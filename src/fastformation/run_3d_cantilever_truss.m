% RUN_3D_CANTILEVER_TRUSS
% Enhanced 3D Space-Frame Cantilever Benchmark (101,400 DOFs)
% Length along X [0, 1.2], Width along Y [0, 0.6], Height along Z [0, 0.6]
% Clamped at X=0, Downward load at tip center (X=L, Y=W/2, Z=H/2)
clear; clc; close all;

nelx = 48; nely = 24; nelz = 24;
L = 1.2; w = 0.6; h = 0.6;
hx = L / nelx; hy = w / nely; hz = h / nelz; % Perfect cubes: 0.025m each
volfrac = 0.18; penal = 3.0; rmin = 1.6; max_iter = 25;

fprintf('========================================================================\n');
fprintf('  ENHANCED 3D SPACE-FRAME CANTILEVER OPTIMIZATION (101,400 DOFs)\n');
fprintf('  Domain: [%.2f x %.2f x %.2f] | Elements: %dx%dx%d (%d elements)\n', ...
    L, w, h, nelx, nely, nelz, nelx*nely*nelz);
fprintf('  Target VolFrac: %.2f | SIMP Penal: %.1f | Filter Radius: %.3f\n', ...
    volfrac, penal, rmin*mean([hx, hy, hz]));
fprintf('========================================================================\n\n');

%% 1. Geometry: X = Length, Y = Width, Z = Height
p = 2;
sp = iga_space_box([0 L; 0 w; 0 h], [nelx, nely, nelz], p);
conn_e = sp.connectivity;
nel = sp.nel; nsh = sp.nsh; ndof = sp.ndof;

% Clamped at X = 0 (root face)
ncp_dir = sp.ndof_dir;
[i1, i2, i3] = ind2sub(ncp_dir, 1:sp.ndof_sc);
clamped_sc = find(i1 == 1);
ndof_sc = sp.ndof_sc;
clamped_dofs = [clamped_sc, clamped_sc + ndof_sc, clamped_sc + 2*ndof_sc];
free_dofs = setdiff(1:ndof, clamped_dofs);
free_mask = false(ndof, 1); free_mask(free_dofs) = true;

% External load: Downward in Z at center of tip face (X = L, Y = w/2, Z = h/2)
mid_y = round(ncp_dir(2)/2);
mid_z = round(ncp_dir(3)/2);
load_sc = find(i1 == ncp_dir(1) & abs(i2 - mid_y) <= 1 & abs(i3 - mid_z) <= 1);
F = zeros(ndof, 1);
load_dofs_z = load_sc + 2*ndof_sc; % Z-displacement DOFs!
F(load_dofs_z) = -1.0 / numel(load_dofs_z);

%% 2. Fast Template Precomputation
t_tmpl = tic;
[Ke_types, type_id] = iga_elasticity_element_matrices(sp, 0.5769, 0.3846);
Ke_types = single(Ke_types);
vals_e = Ke_types(:, :, type_id);
fprintf('Template Precomputation: %.2f s\n', toc(t_tmpl));

%% 3. Fast 3D Filter
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
fprintf('3D Filter Precomputation: %.2f s\n', toc(t_filt));

%% 4. GPU Matrix-Free Allocation
conn_gpu = gpuArray(int32(conn_e));
vals_gpu = gpuArray(vals_e);
F_gpu = gpuArray(single(F));
free_mask_gpu = gpuArray(free_mask);
u_state = zeros(ndof, 1, 'like', F_gpu);

xPhys = volfrac * ones(nel, 1, 'single');
Emin = single(1e-3);
compliance_hist = zeros(max_iter, 1);
time_hist = zeros(max_iter, 1);

fprintf('\nStarting 3D Optimization (%d iterations on GPU)...\n', max_iter);
t_start = tic;

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
    
    for pcg_it = 1:80
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
    
    % Sensitivity Analysis
    ue = u_state(conn_gpu);
    ke_ue = pagemtimes(vals_gpu, reshape(ue, [nsh, 1, nel]));
    Ee = sum(ue .* squeeze(ke_ue), 1)';
    c = gather(double(sum(F_gpu .* u_state)));
    compliance_hist(iter) = c;
    
    dC_raw = -penal * (1 - Emin) * (xPhys.^(penal - 1)) .* gather(Ee);
    dC = (H_filter * (xPhys .* dC_raw)) ./ (H_sum .* max(xPhys, 1e-3));
    
    % OC Update
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
    t_iter = toc(t_it);
    time_hist(iter) = t_iter;
    fprintf('Iter %2d: Compliance = %9.2f | Vol = %.3f | Change = %.4f | Time = %.3f s\n', ...
        iter, c, mean(xPhys), max(abs(xPhys - x_old)), t_iter);
end

t_tot = toc(t_start);
fprintf('\nOptimization Complete in %.2f s (Avg: %.3f s/iter)\n', t_tot, mean(time_hist));

%% 5. Render Beautiful 3D Publication Visualization
save('cantilever_truss_100k_results.mat', 'xPhys', 'compliance_hist', 'time_hist', 'nelx', 'nely', 'nelz', 'L', 'w', 'h');

fig = figure('Color', 'w', 'Position', [80, 80, 1200, 500], 'Visible', 'off');

% Left: 3D Isometric View with High-Fidelity Directional Lighting
subplot(1, 2, 1);
rho_3d = reshape(double(xPhys), [nelx, nely, nelz]);
% Permute for MATLAB meshgrid convention [Y, X, Z]:
rho_perm = permute(rho_3d, [2, 1, 3]); % [nely, nelx, nelz]
rho_smooth = smooth3(rho_perm, 'box', 3);

[X_mesh, Y_mesh, Z_mesh] = meshgrid(linspace(0, L, nelx), linspace(0, w, nely), linspace(0, h, nelz));

% Isosurface at threshold
iso_val = 0.25;
p_iso = patch(isosurface(X_mesh, Y_mesh, Z_mesh, rho_smooth, iso_val));
isonormals(X_mesh, Y_mesh, Z_mesh, rho_smooth, p_iso);

% Set sleek aerospace composite material appearance
set(p_iso, 'FaceColor', [0.18, 0.48, 0.86], 'EdgeColor', 'none', ...
    'SpecularStrength', 0.4, 'SpecularExponent', 25, 'DiffuseStrength', 0.8, 'AmbientStrength', 0.35);

% Professional dual-light illumination
light('Position', [1.5, -1.0, 2.0], 'Style', 'infinite'); % Primary key light from top-front
light('Position', [-1.0, 1.5, 0.5], 'Style', 'infinite');  % Fill light from back-bottom
lighting gouraud;

view([-38, 22]); % Classic aerospace 3/4 isometric perspective
axis equal; axis tight;
xlim([0, L]); ylim([0, w]); zlim([0, h]);
box on; grid on;

set(gca, 'Color', 'w', 'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2], 'ZColor', [0.2 0.2 0.2], ...
    'GridColor', [0.85 0.85 0.85], 'FontSize', 10.5, 'LineWidth', 1.2);
xlabel('Length X [m]', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('Width Y [m]', 'FontWeight', 'bold', 'FontSize', 11);
zlabel('Height Z [m]', 'FontWeight', 'bold', 'FontSize', 11);
title(sprintf('3D Optimal Space-Frame Cantilever (101,400 DOFs, V_f = %.2f)', volfrac), ...
    'FontSize', 11.5, 'FontWeight', 'bold', 'Color', 'k');

% Right: Convergence History
subplot(1, 2, 2);
yyaxis left;
plot(1:max_iter, compliance_hist, 'b-o', 'LineWidth', 2.2, 'MarkerSize', 5, 'MarkerFaceColor', 'b');
ylabel('Compliance J = F^T U', 'FontSize', 11.5, 'FontWeight', 'bold', 'Color', 'b');
set(gca, 'YColor', 'b');

yyaxis right;
semilogy(1:max_iter, abs(diff([compliance_hist(1); compliance_hist])), 'r--s', ...
    'LineWidth', 1.8, 'MarkerSize', 5, 'MarkerFaceColor', 'r');
ylabel('Step Change | \Delta J |', 'FontSize', 11.5, 'FontWeight', 'bold', 'Color', 'r');
set(gca, 'YColor', 'r');

grid on;
set(gca, 'FontSize', 10.5, 'XColor', 'k', 'Color', 'w', 'Box', 'on', ...
    'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
title('3D Convergence History (101,400 DOFs)', 'FontSize', 11.5, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Optimization Iteration', 'FontWeight', 'bold', 'FontSize', 11);

exportgraphics(fig, 'fig_topopt_3d_cantilever.png', 'Resolution', 300);
fprintf('Saved high-resolution 3D figure as fig_topopt_3d_cantilever.png\n');

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
