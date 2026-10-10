function [xPhys, compliance_hist, time_hist, total_time] = topopt_iga_3d(nelx, nely, nelz, volfrac, penal, rmin, max_iter, solver_type)
% TOPOPT_IGA_3D 3D Isogeometric Topology Optimization
% Supports both CPU and GPU Matrix-Free solvers in FP64 and FP32.
%
% Inputs:
%   nelx, nely, nelz - Mesh elements in x, y, z (default: 16, 8, 4)
%   volfrac          - Target volume fraction (default: 0.3)
%   penal            - SIMP exponent (default: 3.0)
%   rmin             - Filter radius in element units (default: 1.5)
%   max_iter         - Maximum iterations (default: 40)
%   solver_type      - 'gpu_mf_fp32' (default), 'gpu_mf_fp64', or 'cpu'

if nargin < 1 || isempty(nelx), nelx = 16; end
if nargin < 2 || isempty(nely), nely = 8; end
if nargin < 3 || isempty(nelz), nelz = 4; end
if nargin < 4 || isempty(volfrac), volfrac = 0.3; end
if nargin < 5 || isempty(penal), penal = 3.0; end
if nargin < 6 || isempty(rmin), rmin = 1.5; end
if nargin < 7 || isempty(max_iter), max_iter = 40; end
if nargin < 8 || isempty(solver_type), solver_type = 'gpu_mf_fp32'; end

fprintf('========================================================================\n');
fprintf('  3D ISOGEOMETRIC TOPOLOGY OPTIMIZATION\n');
fprintf('  Mesh: %dx%dx%d (%d elements) | VolFrac: %.2f | Solver: %s\n', ...
    nelx, nely, nelz, nelx*nely*nelz, volfrac, solver_type);
fprintf('========================================================================\n\n');

%% 1. Geometry and 3D Spline Spaces
L = 1.2; h = 0.6; w = 0.3;
E0 = 1.0; nu = 0.3;
lambda = nu * E0 / ((1 + nu) * (1 - 2 * nu));
mu = E0 / (2 * (1 + nu));

p = 2;
sp = iga_space_box([0 L; 0 h; 0 w], [nelx, nely, nelz], p);

% Boundary DOFs: Clamped at x = 0
ncp_dir = sp.ndof_dir;
[i1, i2, i3] = ind2sub(ncp_dir, 1:sp.ndof_sc);
clamped_sc = find(i1 == 1);
ndof_sc = sp.ndof_sc;
clamped_dofs = [clamped_sc, clamped_sc + ndof_sc, clamped_sc + 2*ndof_sc];
free_dofs = setdiff(1:sp.ndof, clamped_dofs);
free_mask = false(sp.ndof, 1); free_mask(free_dofs) = true;

% External load: Downward (y-direction) load at bottom-center of x = L face
load_sc = find(i1 == ncp_dir(1) & abs(i2 - 1) <= 1 & abs(i3 - ncp_dir(3)/2) <= 1);
F = zeros(sp.ndof, 1);
load_dofs_y = load_sc + ndof_sc;
F(load_dofs_y) = -1.0 / numel(load_dofs_y);

%% 2. Precompute 3D Element Operators
t_pre = tic;
[Ke, type_id] = iga_elasticity_element_matrices(sp, lambda, mu);
nel = sp.nel;
nsh = sp.nsh;
conn_e = sp.connectivity; % [nsh x nel]
vals_e = Ke(:, :, type_id);
[rows_all, cols_all] = iga_element_rows_cols(conn_e);
vals0_all = vals_e(:);
n_per_el = nsh^2;
t_precomp = toc(t_pre);
fprintf('3D Element Precomputation: %.2f s (%d DOFs, %d elements)\n', ...
    t_precomp, sp.ndof, nel);

%% 3. Precompute 3D Sensitivity Filter
% Element centers in 3D
hx = L / nelx; hy = h / nely; hz = w / nelz;
[cx, cy, cz] = ndgrid((0.5:nelx)*hx, (0.5:nely)*hy, (0.5:nelz)*hz);
cx = cx(:); cy = cy(:); cz = cz(:);

% Fast 3D neighborhood search
r_phys = rmin * mean([hx, hy, hz]);
[i_idx, j_idx, dist_val] = rangesearch_3d(cx, cy, cz, r_phys);
H_weights = max(0, r_phys - dist_val);
H_filter = sparse(i_idx, j_idx, H_weights, nel, nel);
H_sum = full(sum(H_filter, 2));

%% 4. GPU Setup for Matrix-Free Solvers
use_gpu = startsWith(solver_type, 'gpu');
is_fp32 = contains(solver_type, 'fp32');

if use_gpu
    if is_fp32
        prec_fn = @single;
    else
        prec_fn = @double;
    end
    conn_gpu = gpuArray(int32(conn_e));
    vals_gpu = gpuArray(prec_fn(vals_e));
    F_gpu = gpuArray(prec_fn(F));
    free_mask_gpu = gpuArray(free_mask);
    H_filter_gpu = gpuArray(prec_fn(H_filter));
    H_sum_gpu = gpuArray(prec_fn(H_sum));
    u_state = zeros(sp.ndof, 1, 'like', F_gpu);
else
    u_state = zeros(sp.ndof, 1);
end

%% 5. Optimization Loop
xPhys = volfrac * ones(nel, 1);
Emin = 1e-3;
compliance_hist = zeros(max_iter, 1);
time_hist = zeros(max_iter, 1);
t_total_start = tic;

fprintf('\n%-6s %-16s %-14s %-14s %-12s\n', ...
    'Iter', 'Compliance J', 'VolFrac', 'Delta_rho_max', 'Time [s]');
fprintf('%s\n', repmat('-', 1, 68));

for iter = 1:max_iter
    t_it = tic;
    x_old = xPhys;
    scale = Emin + (1 - Emin) * (xPhys.^penal);
    
    % State solve
    if use_gpu
        scale_gpu = gpuArray(prec_fn(scale));
        % Compute diagonal preconditioner on-the-fly
        diag_e = zeros(nsh, nel, 'like', scale_gpu);
        for a = 1:nsh
            diag_e(a, :) = squeeze(vals_gpu(a, a, :))' .* scale_gpu';
        end
        K_diag = accumarray(conn_gpu(:), diag_e(:), [sp.ndof, 1]);
        M_inv_gpu = 1 ./ max(K_diag, prec_fn(1e-6));
        
        % Matrix-Free PCG with Warm Start
        matvec = @(p) eval_matvec_3d(p, conn_gpu, vals_gpu, scale_gpu, free_mask_gpu, sp.ndof, nsh, nel);
        r = (F_gpu - matvec(u_state)) .* free_mask_gpu;
        z = M_inv_gpu .* r;
        p_vec = z;
        rz_old = sum(r .* z);
        tol = prec_fn(1e-4);
        norm_f = norm(F_gpu(free_dofs));
        
        for pcg_it = 1:120
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
        
        % Sensitivity Evaluation on GPU
        ue = u_state(conn_gpu);
        ke_ue = pagemtimes(vals_gpu, reshape(ue, [nsh, 1, nel]));
        Ee = sum(ue .* squeeze(ke_ue), 1)'; % [nel x 1] strain energy
        dC_raw = -penal * (1 - Emin) * (xPhys.^(penal - 1)) .* gather(double(Ee));
        c = gather(double(sum(F_gpu .* u_state)));
        
    else % CPU Direct Solver
        K = sparse(rows_all, cols_all, vals0_all .* repelem(scale, n_per_el), sp.ndof, sp.ndof);
        K = 0.5 * (K + K');
        u_state = zeros(sp.ndof, 1);
        u_state(free_dofs) = K(free_dofs, free_dofs) \ F(free_dofs);
        
        % Strain energy on CPU
        ue = u_state(conn_e);
        ke_ue = pagemtimes(vals_e, reshape(ue, [nsh, 1, nel]));
        Ee = sum(ue .* squeeze(ke_ue), 1)';
        dC_raw = -penal * (1 - Emin) * (xPhys.^(penal - 1)) .* Ee;
        c = sum(F .* u_state);
    end
    
    % Sensitivity Filtering
    dC = (H_filter * (xPhys .* dC_raw)) ./ (H_sum .* max(xPhys, 1e-3));
    
    % Optimality Criteria Bisection Update
    l1 = 0; l2 = max(abs(dC(:))) * 2.0; move = 0.2;
    while (l2 - l1) / (l1 + l2 + 1e-10) > 1e-4
        lmid = 0.5 * (l1 + l2);
        Be = (-dC / lmid).^0.5;
        x_cand = max(0.001, max(x_old - move, min(1.0, min(x_old + move, x_old .* Be))));
        if mean(x_cand) > volfrac
            l1 = lmid;
        else
            l2 = lmid;
        end
    end
    
    xPhys = x_cand;
    change = max(abs(xPhys(:) - x_old(:)));
    compliance_hist(iter) = abs(c);
    time_hist(iter) = toc(t_it);
    
    fprintf('%-6d %-16.4f %-14.4f %-14.4e %-12.3f\n', ...
        iter, compliance_hist(iter), mean(xPhys), change, time_hist(iter));
    
    if change < 0.01 && iter >= 15
        fprintf('Convergence achieved at iteration %d!\n', iter);
        break;
    end
end

total_time = toc(t_total_start);
compliance_hist = compliance_hist(1:iter);
time_hist = time_hist(1:iter);

fprintf('%s\n', repmat('=', 1, 68));
fprintf('3D Optimization finished in %d iterations (Total: %.2f s, Avg: %.3f s/iter)\n\n', ...
    iter, total_time, mean(time_hist));

%% 6. Generate 3D Isosurface Visualization
fig = figure('Position', [100, 100, 950, 480], 'Color', 'w', 'InvertHardcopy', 'off');

subplot(1, 2, 1);
rho_3d = reshape(xPhys, [nelx, nely, nelz]);
% Generate smooth grid for isosurface rendering
[X, Y, Z] = meshgrid(linspace(0, L, nelx), linspace(0, h, nely), linspace(0, w, nelz));
rho_plot = permute(rho_3d, [2, 1, 3]);

p_iso = patch(isosurface(X, Y, Z, rho_plot, 0.4));
isonormals(X, Y, Z, rho_plot, p_iso);
set(p_iso, 'FaceColor', [0.2 0.45 0.8], 'EdgeColor', 'none');
daspect([1 1 1]);
view(3); axis tight; axis equal;
camlight('headlight'); lighting gouraud;
box on; grid on;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k', 'FontSize', 10, 'LineWidth', 1.2);
xlabel('X', 'FontWeight', 'bold'); ylabel('Y', 'FontWeight', 'bold'); zlabel('Z', 'FontWeight', 'bold');
title(sprintf('3D Optimal Topology (%dx%dx%d elements, GPU Matrix-Free %s)', nelx, nely, nelz, upper(strrep(solver_type, 'gpu_mf_', ''))), ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k', 'Interpreter', 'none');

subplot(1, 2, 2);
yyaxis left;
plot(1:iter, compliance_hist, 'b-o', 'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'b');
ylabel('Compliance J = F^T U', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'b');
set(gca, 'YColor', 'b');

yyaxis right;
semilogy(1:iter, abs(diff([compliance_hist(1); compliance_hist])), 'r--s', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'r');
ylabel('Step Change | \Delta J |', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r');
set(gca, 'YColor', 'r');

grid on;
set(gca, 'FontSize', 10, 'XColor', 'k', 'Color', 'w', 'Box', 'on', 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
title('3D Convergence History', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Iteration', 'FontWeight', 'bold');

fig_name = sprintf('fig_topopt_3d_cantilever.png');
try exportgraphics(fig, fig_name, 'Resolution', 300); catch, saveas(fig, fig_name); end
fprintf('Saved 3D topology visualization as %s\n', fig_name);

end

function y = eval_matvec_3d(p, conn, vals, scale, free_mask, ndof, nsh, nel)
    p_act = p .* free_mask;
    pe = p_act(conn); % [nsh x nel]
    pe_reshaped = reshape(pe, [nsh, 1, nel]);
    ye = pagemtimes(vals, pe_reshaped);
    ye_scaled = ye .* reshape(scale, [1, 1, nel]);
    y_full = accumarray(conn(:), ye_scaled(:), [ndof, 1]);
    y = y_full .* free_mask;
end

function [i_idx, j_idx, dist_val] = rangesearch_3d(cx, cy, cz, r)
    nel = numel(cx);
    % Block-based neighbor search
    [I, J] = ndgrid(1:nel, 1:nel);
    dx = cx(I) - cx(J);
    dy = cy(I) - cy(J);
    dz = cz(I) - cz(J);
    dist2 = dx.^2 + dy.^2 + dz.^2;
    mask = dist2 <= (r^2);
    i_idx = I(mask);
    j_idx = J(mask);
    dist_val = sqrt(dist2(mask));
end
