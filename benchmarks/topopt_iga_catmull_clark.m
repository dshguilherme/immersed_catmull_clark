function [xPhys, compliance_history, change_history, time_history] = topopt_iga_catmull_clark(nelx, nely, volfrac, penal, rmin, max_iter, use_gpu)
% TOPOPT_IGA_CATMULL_CLARK 2D Cantilever Topology Optimization
% using Catmull-Clark limit basis projection and FastFormation matrix-free GPU operator.
%
% Reproduces the benchmark in FastFormation_TopOpt:
%   Cantilever beam [0, L] x [0, h] with L = 1.0, h = 0.5
%   80 x 40 elements, Catmull-Clark cubic limit basis (p = 3, C^2 continuity)
%   Clamped at left edge (x = 0), downward load at (L, 0.25) / center-right
%   Target volume fraction: 0.5, penal = 3.0, rmin = 2.0, max_iter = 50.

if nargin < 1 || isempty(nelx), nelx = 80; end
if nargin < 2 || isempty(nely), nely = 40; end
if nargin < 3 || isempty(volfrac), volfrac = 0.5; end
if nargin < 4 || isempty(penal), penal = 3.0; end
if nargin < 5 || isempty(rmin), rmin = 2.0; end
if nargin < 6 || isempty(max_iter), max_iter = 50; end
if nargin < 7 || isempty(use_gpu), use_gpu = true; end

% Paths
addpath(genpath('C:\Users\dshgu\OneDrive\Documents\geopdes-master'));
addpath('C:\Users\dshgu\OneDrive\Documents\FastFormation');
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src', 'catmull_clark'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src', 'fastformation'));

fprintf('========================================================================\n');
fprintf('  CATMULL-CLARK IGA TOPOLOGY OPTIMIZATION (FASTFORMATION GPU PCG)\n');
fprintf('  Domain: [0, 1.0] x [0, 0.5] | Mesh: %d x %d | Catmull-Clark Cubic Basis (C^2)\n', nelx, nely);
fprintf('  Target VolFrac: %.2f | SIMP Penal: %.1f | Filter Radius: %.1f | MaxIter: %d\n', ...
    volfrac, penal, rmin, max_iter);
fprintf('========================================================================\n\n');

%% 1. Problem Geometry and Basis Spaces
L = 1.0; h = 0.5;
degree = 3;
problem_data = cantilever_beam(L, h);
method_data.degree     = [degree, degree];
method_data.regularity = [degree-1, degree-1];
method_data.nsub       = [nelx, nely];
method_data.nquad      = [degree+1, degree+1];

[geometry, msh, space] = buildSpaces(problem_data, method_data);
[free_dofs, ~] = grab_cantilever_dofs(space);
free_mask = false(space.ndof, 1);
free_mask(free_dofs) = true;

% External Force Vector (normalized point load on center-right edge)
F = op_f_v_tp(space, msh, problem_data.f);
Fy_tot = abs(sum(F(space.scalar_spaces{1}.ndof + 1 : end)));
if Fy_tot > 0
    F = F / Fy_tot;
end

%% 2. FastFormation Precomputed Element Stiffness Operator
t_pre = tic;
sp_col = sp_precompute(space, msh, 'gradient', true, 'divergence', true);
msh_col = msh_precompute(msh);
l_val = problem_data.lambda_lame(0, 0) * ones(msh.nqn, msh.nel);
m_val = problem_data.mu_lame(0, 0) * ones(msh.nqn, msh.nel);
[rows, cols, vals0] = op_su_ev(sp_col, sp_col, msh_col, l_val, m_val);

nel = msh.nel;
nsh = sp_col.nsh_max;
rows_e = reshape(rows, [nsh, nsh, nel]);
vals_e = reshape(vals0, [nsh, nsh, nel]);
conn_e = squeeze(rows_e(:, 1, :)); % [nsh x nel]

n_per_el = nsh^2;
t_pre_time = toc(t_pre);
fprintf('FastFormation operator ready in %.2f s (DOFs: %d, Elements: %d)\n\n', ...
    t_pre_time, space.ndof, nel);

%% 3. Catmull-Clark Limit Projection Matrices
% On regular quad topology, Catmull-Clark cubic limit basis evaluates via
% tensor-product 1D cubic B-spline projection matrices P1, P2
[P1, P2] = catmull_clark_projection_matrices(nelx, nely);
ncp_x = size(P1, 2);
ncp_y = size(P2, 2);
n_vars = ncp_x * ncp_y;

% Initial uniform distribution for control points
xPhys = volfrac * ones(ncp_x, ncp_y);
V_target = volfrac * (nelx * nely);
V_eval = @(x) sum(sum(P1 * x * (P2')));

Emin = 1e-3;

%% 4. GPU Setup for Matrix-Free PCG
has_gpu = use_gpu && canUseGPU();
if has_gpu
    conn_gpu = gpuArray(int32(conn_e));
    vals_gpu = gpuArray(single(vals_e));
    F_gpu = gpuArray(single(F));
    free_mask_gpu = gpuArray(free_mask);
    u_state = zeros(space.ndof, 1, 'single', 'gpuArray');
    fprintf('Accelerating with GPU: %s\n\n', gpuDevice().Name);
else
    fprintf('Running on CPU solver.\n\n');
end

compliance_history = zeros(max_iter, 1);
change_history = zeros(max_iter, 1);
time_history = zeros(max_iter, 1);

fprintf('%-6s %-16s %-14s %-14s %-12s\n', ...
    'Iter', 'Compliance', 'VolFrac', 'Change', 'Time (s)');
fprintf('%s\n', repmat('-', 1, 68));

change = 1.0;
iter = 0;
move = 0.2;

%% 5. Optimization Loop (Optimality Criteria with Catmull-Clark Adjoint)
while iter < max_iter && change > 1e-3
    iter = iter + 1;
    t_iter = tic;
    x_old = xPhys;
    
    % --- Step 1: Catmull-Clark Basis Projection to Physical Element Densities ---
    rho_grid = P1 * xPhys * (P2');
    rho_e = rho_grid(:);
    
    % --- Step 2: State Solution (GPU Matrix-Free PCG or CPU Direct) ---
    scale = Emin + (1 - Emin) * (rho_e .^ penal);
    
    if has_gpu
        scale_gpu = gpuArray(single(scale));
        
        % Diagonal Jacobi preconditioner on GPU
        diag_e = zeros(nsh, nel, 'single', 'gpuArray');
        for a = 1:nsh
            diag_e(a, :) = squeeze(vals_gpu(a, a, :))' .* scale_gpu';
        end
        K_diag = accumarray(conn_gpu(:), diag_e(:), [space.ndof, 1]);
        M_inv_gpu = 1 ./ max(K_diag, single(1e-6));
        
        % Matrix-Free PCG with Warm Start
        matvec = @(p) eval_matvec_2d(p, conn_gpu, vals_gpu, scale_gpu, free_mask_gpu, space.ndof, nsh, nel);
        r = (F_gpu - matvec(u_state)) .* free_mask_gpu;
        z = M_inv_gpu .* r;
        p_vec = z;
        rz_old = sum(r .* z);
        tol = single(1e-4);
        norm_f = norm(F_gpu(free_dofs));
        
        for pcg_it = 1:200
            Ap = matvec(p_vec);
            pAp = sum(p_vec .* Ap);
            if abs(pAp) < 1e-12, break; end
            alpha = rz_old / pAp;
            u_state = u_state + alpha * p_vec;
            r = r - alpha * Ap;
            if norm(r) / norm_f < tol, break; end
            z = M_inv_gpu .* r;
            rz_new = sum(r .* z);
            p_vec = z + (rz_new / rz_old) * p_vec;
            rz_old = rz_new;
        end
        
        U = double(gather(u_state));
        c = double(gather(sum(F_gpu .* u_state)));
        
        % GPU Element strain energy
        ue = u_state(conn_gpu);
        ke_ue = pagemtimes(vals_gpu, reshape(ue, [nsh, 1, nel]));
        Ee = sum(ue .* squeeze(ke_ue), 1)';
        ce = double(gather(Ee));
    else
        scale_all = repelem(scale, n_per_el);
        K = sparse(rows, cols, vals0 .* scale_all, space.ndof, space.ndof);
        U = zeros(space.ndof, 1);
        U(free_dofs) = K(free_dofs, free_dofs) \ F(free_dofs);
        c = F' * U;
        ce = sum(reshape(U(rows) .* vals0 .* U(cols), n_per_el, nel), 1)';
    end
    
    compliance_history(iter) = c;
    
    % --- Step 3: Sensitivity Formulation ---
    dC_elem = -penal * (1 - Emin) * (rho_e .^ (penal - 1)) .* ce;
    
    % --- Step 4: Adjoint Backpropagation to Catmull-Clark Control Points ---
    % \nabla_{x_cp} J = P1' * (\nabla_{\rho} J) * P2
    dC_cp = P1' * reshape(dC_elem, [nelx, nely]) * P2;
    sens = -dC_cp;
    
    % --- Step 5: Optimality Criteria Bisection Search ---
    l1 = 0;
    l2 = max(sens(:)) * 2;
    while V_eval(max(1e-3, max(x_old - move, min(1.0, min(x_old + move, x_old .* sqrt(max(0, sens / l2))))))) > V_target
        l2 = l2 * 2;
    end
    
    while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4
        lmid = 0.5 * (l1 + l2);
        Be = max(0, sens / lmid);
        x_new = max(1e-3, max(x_old - move, min(1.0, min(x_old + move, x_old .* sqrt(Be)))));
        if V_eval(x_new) > V_target
            l1 = lmid;
        else
            l2 = lmid;
        end
    end
    
    change = max(abs(x_new(:) - x_old(:)));
    xPhys = x_new;
    curr_vol = V_eval(xPhys) / (nelx * nely);
    
    change_history(iter) = change;
    time_history(iter) = toc(t_iter);
    
    fprintf('%-6d %-16.4f %-14.4f %-14.4e %-12.3f\n', ...
        iter, c, curr_vol, change, time_history(iter));
end

fprintf('%s\n', repmat('=', 1, 68));
fprintf('Optimization completed in %d iterations (Total time: %.2f s)\n\n', ...
    iter, sum(time_history(1:iter)));

compliance_history = compliance_history(1:iter);
change_history = change_history(1:iter);
time_history = time_history(1:iter);

%% 6. Generate Publication-Quality Figure (Identical Style to Article)
figDir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end

fig = figure('Position', [100, 100, 1000, 420], 'Color', 'w', 'InvertHardcopy', 'off');

% Left: Optimal Structural Topology
subplot(1, 2, 1);
plot_density = (P1 * xPhys * (P2'))';

imagesc([0 L], [0 h], plot_density);
colormap(flipud(gray));
axis equal tight;
set(gca, 'YDir', 'normal', 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'Box', 'on', 'LineWidth', 1.2);
try clim([0 1]); catch, caxis([0 1]); end
cb = colorbar;
set(cb, 'Color', 'k', 'FontSize', 10);
ylabel(cb, 'Physical Density \rho', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
title(sprintf('Optimal Topology (Catmull-Clark, p = %d, %dx%d)', degree, nelx, nely), ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
xlabel('x / L', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('y / h', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

% Right: Convergence History (Dual Y-Axis matching article)
subplot(1, 2, 2);
yyaxis left;
plot(1:iter, compliance_history, 'b-o', 'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'b');
ylabel('Compliance J = F^T U', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'b');
set(gca, 'YColor', 'b');

yyaxis right;
semilogy(1:iter, change_history, 'r--s', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'r');
ylabel('Max Change \Delta \rho_{max}', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r');
set(gca, 'YColor', 'r');

grid on;
set(gca, 'FontSize', 11, 'XColor', 'k', 'Color', 'w', 'Box', 'on', 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
title('Convergence History (Compliance & Change)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Iteration', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

figPath = fullfile(figDir, 'fig_topopt_catmull_clark_p3.png');
exportgraphics(fig, figPath, 'Resolution', 150);
close(fig);
fprintf('Saved reproduced Catmull-Clark cantilever topology figure: %s\n', figPath);

end

function tf = canUseGPU()
    try
        g = gpuDevice();
        tf = g.DeviceSupported;
    catch
        tf = false;
    end
end

function y = eval_matvec_2d(p, conn, vals, scale, free_mask, ndof, nsh, nel)
    p_act = p .* free_mask;
    pe = p_act(conn);
    pe_reshaped = reshape(pe, [nsh, 1, nel]);
    ye = pagemtimes(vals, pe_reshaped);
    ye_scaled = ye .* reshape(scale, [1, 1, nel]);
    y_full = accumarray(conn(:), ye_scaled(:), [ndof, 1]);
    y = y_full .* free_mask;
end
