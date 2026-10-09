function [u_assembly, contact_results] = solve_assembly_contact_3d(assembly, F_ext, bc_spec, opts)
% SOLVE_ASSEMBLY_CONTACT_3D
% Non-linear unilateral contact solver and bonded interface solver for
% multi-body CAD assemblies on independent non-conforming adaptive octrees.
%
% Implements a semi-smooth Newton active-set loop for unilateral contact:
%   g_n >= 0,  p_n <= 0,  p_n * g_n = 0
% and a direct linear solve for bonded / welded contact interfaces.
%
% Inputs:
%   assembly        - Multi-body struct from setup_assembly_3d
%   F_ext           - External load vector of size [total_dof x 1] (or cell array per body)
%   bc_spec         - Boundary condition specifications (cell array or struct array)
%   opts            - (Optional) settings struct:
%                       .mode        - 'unilateral' (default) or 'bonded'
%                       .gamma_c     - Contact penalty parameter (default: 50.0)
%                       .max_iter    - Max Newton iterations (default: 20)
%                       .tol         - Active set / residual tolerance (default: 1e-4)
%                       .friction    - Friction coefficient mu (default: 0.0)
%
% Outputs:
%   u_assembly      - Solution displacement vector across all bodies [total_dof x 1]
%   contact_results - Struct with diagnostics:
%                       .iterations    - Number of Newton iterations
%                       .active_pairs  - Indices of active contact pairs
%                       .contact_press - Normal contact pressure profile
%                       .min_gap       - Minimum normal gap after solution
%                       .converged     - Logical true if converged

if ~isfield(opts, 'mode'), opts.mode = 'unilateral'; end
if ~isfield(opts, 'gamma_c'), opts.gamma_c = 50.0; end
if ~isfield(opts, 'max_iter'), opts.max_iter = 20; end
if ~isfield(opts, 'tol'), opts.tol = 1e-4; end
if ~isfield(opts, 'linear_solver'), opts.linear_solver = 'direct'; end % 'direct' or 'gpu_pcg'
if ~isfield(opts, 'pcg_tol'), opts.pcg_tol = 1e-6; end
if ~isfield(opts, 'pcg_max_iter'), opts.pcg_max_iter = 1000; end

total_dof = assembly.total_dof;

% Format F_ext if provided as cell array per body
if iscell(F_ext)
    F_full = zeros(total_dof, 1);
    for i = 1:assembly.n_bodies
        b = assembly.bodies{i};
        F_full(b.dof_range) = F_ext{i};
    end
    F_ext = F_full;
end

% Extract Dirichlet and Neumann BCs from bc_spec
fixed_dofs = [];
prescribed_vals = [];
F_neumann = zeros(total_dof, 1);
K_bc_extra = sparse(total_dof, total_dof);

if ~isempty(bc_spec)
    if ~iscell(bc_spec) && isstruct(bc_spec), bc_spec = num2cell(bc_spec); end
    
    for k = 1:numel(bc_spec)
        bc = bc_spec{k};
        b_idx = 1;
        if isfield(bc, 'body_idx'), b_idx = bc.body_idx; end
        
        body = assembly.bodies{b_idx};
        mesh = body.mesh;
        brep = body.brep;
        dof_range = body.dof_range;
        
        [K_b, F_b, state_b] = apply_boundary_conditions(mesh, brep, {bc});
        
        % Map to assembly DOFs
        if ~isempty(state_b.fixed_dofs)
            fixed_dofs = [fixed_dofs; dof_range(state_b.fixed_dofs)'];
            prescribed_vals = [prescribed_vals; state_b.prescribed_vals];
        end
        
        F_neumann(dof_range) = F_neumann(dof_range) + F_b;
        K_bc_extra(dof_range, dof_range) = K_bc_extra(dof_range, dof_range) + (K_b - mesh.K_master);
    end
end

if ~isempty(fixed_dofs)
    [fixed_dofs, u_idx] = unique(fixed_dofs);
    prescribed_vals = prescribed_vals(u_idx);
end
free_dofs = setdiff((1:total_dof)', fixed_dofs);

% Build Contact Interpolation Operators for all detected interfaces
% Each interface connects body_A and body_B
n_inter = numel(assembly.interfaces);
all_pairs = [];
pair_cursor = 0;

for int_id = 1:n_inter
    inter = assembly.interfaces{int_id};
    iA = inter.body_A;
    iB = inter.body_B;
    bA = assembly.bodies{iA};
    bB = assembly.bodies{iB};
    
    % Precompute shape function interpolation matrices MA, MB for all contact points
    [MA, MB, valid_pts] = build_interface_projectors(bA, bB, inter);
    
    inter.MA = MA;
    inter.MB = MB;
    inter.valid_pts = valid_pts;
    assembly.interfaces{int_id} = inter;
    
    pair_cursor = pair_cursor + sum(valid_pts);
end

K_base = assembly.K_assembly + K_bc_extra;
F_total = F_ext + F_neumann;

% Mode 1: BONDED INTERFACE (Linear solve)
if strcmpi(opts.mode, 'bonded')
    fprintf('Solving Multi-Body Assembly with BONDED Contact Interfaces...\n');
    [K_contact, F_contact] = assemble_all_contact_matrices(assembly, opts.gamma_c, true);
    
    K_sys = K_base + K_contact;
    F_sys = F_total + F_contact;
    
    u_assembly = solve_coupled_linear_step(assembly, K_sys, F_sys, K_contact, K_bc_extra, ...
                                           fixed_dofs, prescribed_vals, free_dofs, opts);
    
    contact_results.iterations = 1;
    contact_results.converged = true;
    contact_results.mode = 'bonded';
    contact_results.min_gap = 0.0;
    return;
end

% Mode 2: UNILATERAL CONTACT (Semi-smooth Newton active-set loop)
fprintf('Solving Multi-Body Assembly with UNILATERAL Contact (Active-Set Newton)...\n');

u_assembly = zeros(total_dof, 1);
if ~isempty(fixed_dofs)
    u_assembly(fixed_dofs) = prescribed_vals;
end

prev_active = [];
converged = false;

for iter = 1:opts.max_iter
    % Compute gap and active set across all interfaces
    [active_flags, gaps, normal_pressures] = evaluate_interface_states(assembly, u_assembly, opts.gamma_c);
    
    n_act = sum(active_flags);
    min_gap = min(gaps);
    
    fprintf('  Iter %d: %d active contact pairs (Min gap: %+.3e)\n', iter, n_act, min_gap);
    
    % Check active-set convergence
    if iter > 1 && isequal(active_flags, prev_active)
        converged = true;
        fprintf('  --> Active set converged in %d iterations.\n', iter);
        break;
    end
    prev_active = active_flags;
    
    % Assemble active contact stiffness and linearization load
    [K_contact, F_contact] = assemble_active_contact_matrices(assembly, active_flags, opts.gamma_c);
    
    K_sys = K_base + K_contact;
    F_sys = F_total + F_contact;
    
    % Solve linearized system
    u_new = solve_coupled_linear_step(assembly, K_sys, F_sys, K_contact, K_bc_extra, ...
                                      fixed_dofs, prescribed_vals, free_dofs, opts);
    
    du_norm = norm(u_new - u_assembly) / max(1.0, norm(u_new));
    u_assembly = u_new;
    
    if du_norm < opts.tol && iter > 1
        converged = true;
        fprintf('  --> Displacement increment converged (du/u = %.2e) in %d iterations.\n', du_norm, iter);
        break;
    end
end

[active_flags, gaps, normal_pressures] = evaluate_interface_states(assembly, u_assembly, opts.gamma_c);

contact_results.iterations = iter;
contact_results.converged = converged;
contact_results.active_pairs = find(active_flags);
contact_results.n_active = sum(active_flags);
contact_results.contact_press = normal_pressures;
contact_results.min_gap = min(gaps);
contact_results.mode = 'unilateral';

end

% --- HELPER ROUTINES ---

function [MA, MB, valid_pts] = build_interface_projectors(bA, bB, inter)
% Evaluates interpolation matrices MA and MB mapping master DOFs of bA and bB
% to displacements at the contact quadrature points: u(x) = M * u_master
meshA = bA.mesh;
meshB = bB.mesh;
ptsA = inter.pts_A;
ptsB = inter.pts_B;
nP = inter.n_pairs;

b_elemA = meshA.bounds;
b_elemB = meshB.bounds;

valid_pts = true(nP, 1);

% Evaluate shape function matrices
% MA is [3*nP x 3*meshA.n_master], MB is [3*nP x 3*meshB.n_master]
i_A = []; j_A = []; s_A = [];
i_B = []; j_B = []; s_B = [];

xi_c  = [-1,  1,  1, -1, -1,  1,  1, -1];
eta_c = [-1, -1,  1,  1, -1, -1,  1,  1];
zt_c  = [-1, -1, -1, -1,  1,  1,  1,  1];

for p = 1:nP
    pA = ptsA(p, :);
    pB = ptsB(p, :);
    
    % Element in A
    eA = find(pA(1) >= b_elemA(:,1)-1e-5 & pA(1) <= b_elemA(:,2)+1e-5 & ...
              pA(2) >= b_elemA(:,3)-1e-5 & pA(2) <= b_elemA(:,4)+1e-5 & ...
              pA(3) >= b_elemA(:,5)-1e-5 & pA(3) <= b_elemA(:,6)+1e-5, 1);
          
    % Element in B
    eB = find(pB(1) >= b_elemB(:,1)-1e-5 & pB(1) <= b_elemB(:,2)+1e-5 & ...
              pB(2) >= b_elemB(:,3)-1e-5 & pB(2) <= b_elemB(:,4)+1e-5 & ...
              pB(3) >= b_elemB(:,5)-1e-5 & pB(3) <= b_elemB(:,6)+1e-5, 1);
          
    if isempty(eA) || isempty(eB)
        valid_pts(p) = false;
        continue;
    end
    
    % Shape functions in A
    b_eA = b_elemA(eA, :);
    hA = b_eA([2 4 6]) - b_eA([1 3 5]);
    cA = 0.5 * (b_eA([1 3 5]) + b_eA([2 4 6]));
    xiA  = max(-1, min(1, 2 * (pA(1) - cA(1)) / hA(1)));
    etaA = max(-1, min(1, 2 * (pA(2) - cA(2)) / hA(2)));
    ztA  = max(-1, min(1, 2 * (pA(3) - cA(3)) / hA(3)));
    NA = 0.125 * (1 + xi_c * xiA) .* (1 + eta_c * etaA) .* (1 + zt_c * ztA);
    
    enA = meshA.elem_nodes(eA, :);
    for a = 1:8
        n_id = enA(a);
        m_weights = meshA.T(n_id, :);
        m_active = find(m_weights > 1e-4);
        for m_id = m_active(:)'
            w_eff = NA(a) * m_weights(m_id);
            for c = 1:3
                row = (p - 1) * 3 + c;
                col = (m_id - 1) * 3 + c;
                i_A = [i_A; row];
                j_A = [j_A; col];
                s_A = [s_A; w_eff];
            end
        end
    end
    
    % Shape functions in B
    b_eB = b_elemB(eB, :);
    hB = b_eB([2 4 6]) - b_eB([1 3 5]);
    cB = 0.5 * (b_eB([1 3 5]) + b_eB([2 4 6]));
    xiB  = max(-1, min(1, 2 * (pB(1) - cB(1)) / hB(1)));
    etaB = max(-1, min(1, 2 * (pB(2) - cB(2)) / hB(2)));
    ztB  = max(-1, min(1, 2 * (pB(3) - cB(3)) / hB(3)));
    NB = 0.125 * (1 + xi_c * xiB) .* (1 + eta_c * etaB) .* (1 + zt_c * ztB);
    
    enB = meshB.elem_nodes(eB, :);
    for a = 1:8
        n_id = enB(a);
        m_weights = meshB.T(n_id, :);
        m_active = find(m_weights > 1e-4);
        for m_id = m_active(:)'
            w_eff = NB(a) * m_weights(m_id);
            for c = 1:3
                row = (p - 1) * 3 + c;
                col = (m_id - 1) * 3 + c;
                i_B = [i_B; row];
                j_B = [j_B; col];
                s_B = [s_B; w_eff];
            end
        end
    end
end

MA = sparse(i_A, j_A, s_A, 3 * nP, 3 * meshA.n_master);
MB = sparse(i_B, j_B, s_B, 3 * nP, 3 * meshB.n_master);

end

function [active_flags, gaps, normal_pressures] = evaluate_interface_states(assembly, u_assembly, gamma_c)
total_pairs = 0;
for int_id = 1:numel(assembly.interfaces)
    total_pairs = total_pairs + assembly.interfaces{int_id}.n_pairs;
end

active_flags = false(total_pairs, 1);
gaps = zeros(total_pairs, 1);
normal_pressures = zeros(total_pairs, 1);

p_offset = 0;
for int_id = 1:numel(assembly.interfaces)
    inter = assembly.interfaces{int_id};
    bA = assembly.bodies{inter.body_A};
    bB = assembly.bodies{inter.body_B};
    
    uA = u_assembly(bA.dof_range);
    uB = u_assembly(bB.dof_range);
    
    dispA = reshape(inter.MA * uA, 3, inter.n_pairs)';
    dispB = reshape(inter.MB * uB, 3, inter.n_pairs)';
    
    % Current positions
    x_curr_A = inter.pts_A + dispA;
    x_curr_B = inter.pts_B + dispB;
    
    % Normal gap: g_n = (x_B - x_A) . n_A
    % Where n_A is outward normal of body A pointing toward body B
    nA = inter.normals;
    gap_n = sum((x_curr_B - x_curr_A) .* nA, 2);
    
    % Effective mesh size and material modulus
    hA = mean(mean(bA.mesh.h_elem));
    hB = mean(mean(bB.mesh.h_elem));
    heff = 0.5 * (hA + hB);
    EA = max(1.0, mean(diag(bA.mesh.C_tensor(1:3, 1:3))));
    EB = max(1.0, mean(diag(bB.mesh.C_tensor(1:3, 1:3))));
    E_eff = 0.5 * (EA + EB);
    penalty = gamma_c * E_eff / heff;
    
    % Active if penetration occurs (gap_n <= 1e-6)
    is_act = (gap_n <= 1e-6) & inter.valid_pts;
    
    % Contact pressure: p_n = [ -penalty * gap_n ]_+
    p_n = zeros(inter.n_pairs, 1);
    p_n(is_act) = -penalty * gap_n(is_act);
    
    idx_range = (p_offset + 1):(p_offset + inter.n_pairs);
    active_flags(idx_range) = is_act;
    gaps(idx_range) = gap_n;
    normal_pressures(idx_range) = p_n;
    
    p_offset = p_offset + inter.n_pairs;
end

end

function [K_contact, F_contact] = assemble_active_contact_matrices(assembly, active_flags, gamma_c)
total_dof = assembly.total_dof;
K_contact = sparse(total_dof, total_dof);
F_contact = zeros(total_dof, 1);

p_offset = 0;
for int_id = 1:numel(assembly.interfaces)
    inter = assembly.interfaces{int_id};
    bA = assembly.bodies{inter.body_A};
    bB = assembly.bodies{inter.body_B};
    
    idx_range = (p_offset + 1):(p_offset + inter.n_pairs);
    act = active_flags(idx_range);
    p_offset = p_offset + inter.n_pairs;
    
    if ~any(act), continue; end
    
    act_pts = find(act);
    hA = mean(mean(bA.mesh.h_elem));
    hB = mean(mean(bB.mesh.h_elem));
    heff = 0.5 * (hA + hB);
    EA = max(1.0, mean(diag(bA.mesh.C_tensor(1:3, 1:3))));
    EB = max(1.0, mean(diag(bB.mesh.C_tensor(1:3, 1:3))));
    E_eff = 0.5 * (EA + EB);
    penalty = gamma_c * E_eff / heff;
    
    % Assemble for active points
    for p = act_pts(:)'
        Ap = inter.areas(p);
        np = inter.normals(p, :)';
        P_n = np * np'; % Normal projection 3x3
        
        row_3 = (p-1)*3 + (1:3);
        MA_p = inter.MA(row_3, :);
        MB_p = inter.MB(row_3, :);
        
        % Relative jump operator: J = [ -MA_p, MB_p ]
        % K_block = penalty * Ap * J' * P_n * J
        k_val = penalty * Ap;
        
        K_AA = k_val * (MA_p' * P_n * MA_p);
        K_AB = -k_val * (MA_p' * P_n * MB_p);
        K_BA = -k_val * (MB_p' * P_n * MA_p);
        K_BB = k_val * (MB_p' * P_n * MB_p);
        
        rangeA = bA.dof_range;
        rangeB = bB.dof_range;
        
        K_contact(rangeA, rangeA) = K_contact(rangeA, rangeA) + K_AA;
        K_contact(rangeA, rangeB) = K_contact(rangeA, rangeB) + K_AB;
        K_contact(rangeB, rangeA) = K_contact(rangeB, rangeA) + K_BA;
        K_contact(rangeB, rangeB) = K_contact(rangeB, rangeB) + K_BB;
        
        % Initial gap offset force: f = k_val * (x_B - x_A) . n * J' * n
        g0 = dot(inter.pts_B(p,:) - inter.pts_A(p,:), np');
        f_A = (k_val * g0) * (MA_p' * np);
        f_B = -(k_val * g0) * (MB_p' * np);
        
        F_contact(rangeA) = F_contact(rangeA) + f_A;
        F_contact(rangeB) = F_contact(rangeB) + f_B;
    end
end

K_contact = 0.5 * (K_contact + K_contact');
end

function [K_contact, F_contact] = assemble_all_contact_matrices(assembly, gamma_c, is_bonded)
total_dof = assembly.total_dof;
K_contact = sparse(total_dof, total_dof);
F_contact = zeros(total_dof, 1);

for int_id = 1:numel(assembly.interfaces)
    inter = assembly.interfaces{int_id};
    bA = assembly.bodies{inter.body_A};
    bB = assembly.bodies{inter.body_B};
    
    hA = mean(mean(bA.mesh.h_elem));
    hB = mean(mean(bB.mesh.h_elem));
    heff = 0.5 * (hA + hB);
    EA = max(1.0, mean(diag(bA.mesh.C_tensor(1:3, 1:3))));
    EB = max(1.0, mean(diag(bB.mesh.C_tensor(1:3, 1:3))));
    E_eff = 0.5 * (EA + EB);
    penalty = gamma_c * E_eff / heff;
    
    valid_pts = find(inter.valid_pts);
    for p = valid_pts(:)'
        Ap = inter.areas(p);
        np = inter.normals(p, :)';
        if is_bonded
            P_op = eye(3); % Bonded connects all 3 directions
        else
            P_op = np * np';
        end
        
        row_3 = (p-1)*3 + (1:3);
        MA_p = inter.MA(row_3, :);
        MB_p = inter.MB(row_3, :);
        
        k_val = penalty * Ap;
        
        K_AA = k_val * (MA_p' * P_op * MA_p);
        K_AB = -k_val * (MA_p' * P_op * MB_p);
        K_BA = -k_val * (MB_p' * P_op * MA_p);
        K_BB = k_val * (MB_p' * P_op * MB_p);
        
        rangeA = bA.dof_range;
        rangeB = bB.dof_range;
        
        K_contact(rangeA, rangeA) = K_contact(rangeA, rangeA) + K_AA;
        K_contact(rangeA, rangeB) = K_contact(rangeA, rangeB) + K_AB;
        K_contact(rangeB, rangeA) = K_contact(rangeB, rangeA) + K_BA;
        K_contact(rangeB, rangeB) = K_contact(rangeB, rangeB) + K_BB;
        
        g0 = inter.pts_B(p,:) - inter.pts_A(p,:);
        f_A = k_val * (MA_p' * (P_op * g0'));
        f_B = -k_val * (MB_p' * (P_op * g0'));
        
        F_contact(rangeA) = F_contact(rangeA) + f_A;
        F_contact(rangeB) = F_contact(rangeB) + f_B;
    end
end

K_contact = 0.5 * (K_contact + K_contact');
end

function u_out = solve_coupled_linear_step(assembly, K_sys, F_sys, K_contact, K_bc_extra, ...
                                           fixed_dofs, prescribed_vals, free_dofs, opts)
% Solves one coupled linear step either via direct elimination or GPU/PCG
total_dof = assembly.total_dof;
u_out = zeros(total_dof, 1);

if strcmpi(opts.linear_solver, 'gpu_pcg')
    % GPU / Matrix-free PCG path
    if ~isempty(fixed_dofs)
        u_out(fixed_dofs) = prescribed_vals;
        p_fix = zeros(total_dof, 1);
        p_fix(fixed_dofs) = prescribed_vals;
        y_fix = gpu_assembly_matvec(assembly, p_fix, K_contact, K_bc_extra, struct('use_gpu', true));
        b_free = F_sys(free_dofs) - y_fix(free_dofs);
    else
        b_free = F_sys(free_dofs);
    end
    
    % Diagonal Jacobi preconditioner
    diag_K = full(diag(K_sys));
    diag_K(abs(diag_K) < 1e-12) = 1.0;
    inv_diag = 1.0 ./ diag_K(free_dofs);
    
    % PCG iteration
    p_full = zeros(total_dof, 1);
    x_free = zeros(numel(free_dofs), 1);
    r = b_free;
    z = inv_diag .* r;
    p = z;
    rz_old = dot(r, z);
    
    tol = opts.pcg_tol;
    norm_b = norm(b_free);
    if norm_b == 0, norm_b = 1.0; end
    
    for it = 1:opts.pcg_max_iter
        p_full(free_dofs) = p;
        Ap_full = gpu_assembly_matvec(assembly, p_full, K_contact, K_bc_extra, struct('use_gpu', true));
        Ap = Ap_full(free_dofs);
        
        alpha = rz_old / max(dot(p, Ap), 1e-16);
        x_free = x_free + alpha * p;
        r = r - alpha * Ap;
        
        res_norm = norm(r) / norm_b;
        if res_norm < tol
            break;
        end
        
        z = inv_diag .* r;
        rz_new = dot(r, z);
        beta = rz_new / rz_old;
        p = z + beta * p;
        rz_old = rz_new;
    end
    u_out(free_dofs) = x_free;
else
    % Direct backslash linear solve
    if ~isempty(fixed_dofs)
        u_out(fixed_dofs) = prescribed_vals;
        F_free = F_sys(free_dofs) - K_sys(free_dofs, fixed_dofs) * prescribed_vals;
        u_out(free_dofs) = K_sys(free_dofs, free_dofs) \ F_free;
    else
        u_out = K_sys \ F_sys;
    end
end

end
