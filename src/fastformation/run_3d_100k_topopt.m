% RUN_3D_100K_TOPOPT
% 3D Isogeometric Topology Optimization with 101,184 DOFs
% Uses GPU Matrix-Free Solver in Single Precision (FP32)
clear; clc; close all;
nelx = 60; nely = 30; nelz = 15;
L = 1.2; h = 0.6; w = 0.3;
volfrac = 0.3; penal = 3.0; rmin = 1.5; max_iter = 20;

fprintf('========================================================================\n');
fprintf('  LARGE-SCALE 3D ISOGEOMETRIC TOPOLOGY OPTIMIZATION (100k+ DOFs)\n');
fprintf('  Mesh: %dx%dx%d (%d elements) | VolFrac: %.2f | GPU Matrix-Free (FP32)\n', ...
    nelx, nely, nelz, nelx*nely*nelz, volfrac);
fprintf('========================================================================\n\n');

%% 1. Geometry and Spline Spaces
t_setup = tic;
hx = L / nelx; hy = h / nely; hz = w / nelz;
p = 2;
sp = iga_space_box([0 L; 0 h; 0 w], [nelx, nely, nelz], p);
conn_e = sp.connectivity;
nel = sp.nel;
nsh = sp.nsh;
ndof = sp.ndof;
fprintf('Model Discretization: %d elements, %d DOFs (Setup: %.2f s)\n', ...
    nel, ndof, toc(t_setup));

% Boundary conditions: clamped at x = 0 face
ncp_dir = sp.ndof_dir;
[i1, i2, i3] = ind2sub(ncp_dir, 1:sp.ndof_sc);
clamped_sc = find(i1 == 1);
ndof_sc = sp.ndof_sc;
clamped_dofs = [clamped_sc, clamped_sc + ndof_sc, clamped_sc + 2*ndof_sc];
free_dofs = setdiff(1:ndof, clamped_dofs);
free_mask = false(ndof, 1); free_mask(free_dofs) = true;

% External load: Downward at bottom center of x = L face
load_sc = find(i1 == ncp_dir(1) & abs(i2 - 1) <= 1 & abs(i3 - ncp_dir(3)/2) <= 1);
F = zeros(ndof, 1);
load_dofs_y = load_sc + ndof_sc;
F(load_dofs_y) = -1.0 / numel(load_dofs_y);

%% 2. Fast Template Element Precomputation
t_tmpl = tic;
[Ke_types, type_id] = iga_elasticity_element_matrices(sp, 0.5769, 0.3846);
Ke_types = single(Ke_types);
vals_e = Ke_types(:, :, type_id);
fprintf('Element Operators Formed: %.2f s (%d exact element types)\n', toc(t_tmpl), size(Ke_types, 3));

%% 3. Fast 3D Filter Precomputation
t_filt = tic;
r_phys = rmin * mean([hx, hy, hz]);
Rx = ceil(r_phys / hx); Ry = ceil(r_phys / hy); Rz = ceil(r_phys / hz);
[dix, diy, diz] = ndgrid(-Rx:Rx, -Ry:Ry, -Rz:Rz);
d2 = (dix*hx).^2 + (diy*hy).^2 + (diz*hz).^2;
ok = d2 <= r_phys^2;
dix = dix(ok); diy = diy(ok); diz = diz(ok);
d_dist = sqrt(d2(ok));
n_off = numel(dix);
i_list = repmat(1:nel, n_off, 1);
j_list = zeros(n_off, nel);
dist_list = repmat(d_dist, 1, nel);
for k = 1:n_off
    jx = ix + dix(k); jy = iy + diy(k); jz = iz + diz(k);
    in_b = (jx >= 1 & jx <= nelx) & (jy >= 1 & jy <= nely) & (jz >= 1 & jz <= nelz);
    j_lin = sub2ind([nelx, nely, nelz], max(1, min(nelx, jx)), max(1, min(nely, jy)), max(1, min(nelz, jz)));
    j_lin(~in_b) = 0;
    j_list(k, :) = j_lin;
end
keep = (j_list > 0);
H_weights = single(max(0, r_phys - dist_list(keep)));
H_filter = sparse(double(i_list(keep)), double(j_list(keep)), double(H_weights), nel, nel);
H_sum = full(sum(H_filter, 2));
fprintf('3D Filter Precomputed: %.2f s (%d neighbor pairs)\n', toc(t_filt), numel(H_weights));

%% 4. Device Memory Allocation (GPU)
t_dev = tic;
conn_gpu = gpuArray(int32(conn_e));
vals_gpu = gpuArray(vals_e);
F_gpu = gpuArray(single(F));
free_mask_gpu = gpuArray(free_mask);
u_state = zeros(ndof, 1, 'like', F_gpu);
fprintf('GPU Allocation Complete: %.2f s (Device Memory: ~720 MB)\n', toc(t_dev));

%% 5. Topology Optimization Iterations
xPhys = volfrac * ones(nel, 1, 'single');
Emin = single(1e-3);
compliance_hist = zeros(max_iter, 1);
time_hist = zeros(max_iter, 1);

fprintf('\n%-6s %-16s %-14s %-14s %-12s\n', ...
    'Iter', 'Compliance J', 'VolFrac', 'Delta_rho_max', 'Time [s]');
fprintf('%s\n', repmat('-', 1, 68));

t_opt_start = tic;

for iter = 1:max_iter
    t_it = tic;
    x_old = xPhys;
    scale = Emin + (1 - Emin) * (xPhys.^penal);
    scale_gpu = gpuArray(scale);
    
    % Diagonal Preconditioner
    diag_vals = zeros(nsh, nel, 'like', vals_gpu);
    for a = 1:nsh
        diag_vals(a, :) = squeeze(vals_gpu(a, a, :))' .* scale_gpu';
    end
    K_diag = accumarray(conn_gpu(:), diag_vals(:), [ndof, 1]);
    M_inv = 1 ./ max(K_diag, single(1e-6));
    
    % Warm-Started Matrix-Free PCG
    matvec = @(p) eval_matvec_3d(p, conn_gpu, vals_gpu, scale_gpu, free_mask_gpu, ndof);
    r = (F_gpu - matvec(u_state)) .* free_mask_gpu;
    z = M_inv .* r;
    p_vec = z;
    rz_old = sum(r .* z);
    tol = single(1e-4);
    norm_f = norm(F_gpu(free_dofs));
    
    max_pcg = 80;
    for pcg_it = 1:max_pcg
        Ap = matvec(p_vec);
        pAp = sum(p_vec .* Ap);
        if abs(pAp) < 1e-12, break; end
        alpha = rz_old / pAp;
        u_state = u_state + alpha * p_vec;
        r = r - alpha * Ap;
        if norm(r) / norm_f < tol, break; end
        z = M_inv .* r;
        rz_new = sum(r .* z);
        p_vec = z + (rz_new / rz_old) * p_vec;
        rz_old = rz_new;
    end
    
    % Sensitivity Analysis on GPU
    ue = u_state(conn_gpu);
    ke_ue = pagemtimes(vals_gpu, reshape(ue, [nsh, 1, nel]));
    Ee = sum(ue .* squeeze(ke_ue), 1)';
    c = gather(double(sum(F_gpu .* u_state)));
    compliance_hist(iter) = c;
    
    dC_raw = -penal * (1 - Emin) * (xPhys.^(penal - 1)) .* gather(Ee);
    
    % Sensitivity Filter
    dC = (H_filter * (xPhys .* dC_raw)) ./ (H_sum .* max(xPhys, 1e-3));
    
    % Dynamic Bisection Optimality Criteria Update
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
    
    change = max(abs(xPhys - x_old));
    t_iter = toc(t_it);
    time_hist(iter) = t_iter;
    
    fprintf('%-6d %-16.4f %-14.4f %-14.4f %-12.3f\n', ...
        iter, c, mean(xPhys), change, t_iter);
end

t_total_opt = toc(t_opt_start);
fprintf('%s\n', repmat('-', 1, 68));
fprintf('Total Optimization Time (%d iters): %.2f s (Avg: %.3f s/iter)\n\n', ...
    max_iter, t_total_opt, mean(time_hist));

%% 6. Save Data and Render Publication Figure
save('topopt_3d_100k_results.mat', 'xPhys', 'compliance_hist', 'time_hist', 't_total_opt', 'nelx', 'nely', 'nelz', 'ndof');
fprintf('Results saved to topopt_3d_100k_results.mat\n');

fprintf('Rendering 3D Optimal Structural Topology...\n');
fig = figure('Color', 'w', 'Position', [100, 100, 1150, 480], 'Visible', 'off');

subplot(1, 2, 1);
rho_3d = reshape(double(xPhys), [nelx, nely, nelz]);
rho_smooth = smooth3(rho_3d, 'box', 3);

[X_mesh, Y_mesh, Z_mesh] = meshgrid(linspace(0, h, nely), linspace(0, L, nelx), linspace(0, w, nelz));
p_iso = patch(isosurface(X_mesh, Y_mesh, Z_mesh, rho_smooth, 0.35));
isonormals(X_mesh, Y_mesh, Z_mesh, rho_smooth, p_iso);

set(p_iso, 'FaceColor', [0.15, 0.45, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.95);
view(3); axis tight; axis equal;
camlight('headlight'); lighting gouraud;
box on; grid on;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k', 'FontSize', 10, 'LineWidth', 1.2);
xlabel('Y (Height)', 'FontWeight', 'bold'); ylabel('X (Length)', 'FontWeight', 'bold'); zlabel('Z (Width)', 'FontWeight', 'bold');
title(sprintf('3D Optimal Topology (%dx%dx%d, 101,184 DOFs, GPU FP32)', nelx, nely, nelz), ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

subplot(1, 2, 2);
yyaxis left;
plot(1:max_iter, compliance_hist, 'b-o', 'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'b');
ylabel('Compliance J = F^T U', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'b');
set(gca, 'YColor', 'b');

yyaxis right;
semilogy(1:max_iter, abs(diff([compliance_hist(1); compliance_hist])), 'r--s', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'r');
ylabel('Step Change | \Delta J |', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r');
set(gca, 'YColor', 'r');

grid on;
set(gca, 'FontSize', 10, 'XColor', 'k', 'Color', 'w', 'Box', 'on', 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
title('3D Convergence History (101,184 DOFs)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Iteration', 'FontWeight', 'bold');

fig_name = 'fig_topopt_3d_cantilever.png';
try
    exportgraphics(fig, fig_name, 'Resolution', 300);
catch
    saveas(fig, fig_name);
end
fprintf('Saved updated 100k+ DOF 3D topology figure as %s\n', fig_name);

function y = eval_matvec_3d(p, conn, vals, scale, free_mask, ndof)
    p_act = p .* free_mask;
    pe = p_act(conn);
    nsh = size(conn, 1);
    nel = size(conn, 2);
    ye = pagemtimes(vals, reshape(pe, [nsh, 1, nel]));
    ye_scaled = ye .* reshape(scale, [1, 1, nel]);
    y_full = accumarray(conn(:), ye_scaled(:), [ndof, 1]);
    y = y_full .* free_mask;
end
