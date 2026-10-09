% TEST_CENTER_CANTILEVER_3D
% Test 3D Cantilever with 101,400 DOFs, cubic elements (48x24x24),
% clamped at x=0, vertical point load at center of tip (x=L, y=H/2, z=W/2).
clear; clc; close all;
addpath(genpath('C:\Users\dshgu\OneDrive\Documents\geopdes-master'));
addpath('C:\Users\dshgu\OneDrive\Documents\FastFormation');

nelx = 48; nely = 24; nelz = 24;
L = 1.2; h = 0.6; w = 0.6; % Aspect ratio 2:1:1
hx = L / nelx; hy = h / nely; hz = w / nelz; % hx = hy = hz = 0.025 (perfect cubes!)
volfrac = 0.25; penal = 3.0; rmin = 1.8; max_iter = 22;

fprintf('Discretization: %dx%dx%d (%d elements, %d DOFs)\n', ...
    nelx, nely, nelz, nelx*nely*nelz, 3*(nelx+2)*(nely+2)*(nelz+2));

srf = nrb4surf([0 0 0], [L 0 0], [0 h 0], [L h 0]);
vol = nrbextrude(srf, [0 0 w]);

pdata.geo_name = vol;
pdata.drchlt_sides = []; pdata.nmnn_sides = [];
pdata.press_sides = [];  pdata.symm_sides = [];
pdata.E = 1.0; pdata.nu = 0.3;
pdata.lambda_lame = @(x,y,z) 0.5769 * ones(size(x));
pdata.mu_lame     = @(x,y,z) 0.3846 * ones(size(x));

p = 2;
mdata.degree     = [p, p, p];
mdata.regularity = [p-1, p-1, p-1];
mdata.nsub       = [nelx, nely, nelz];
mdata.nquad      = [p+1, p+1, p+1];

[geo, msh, sp] = buildSpaces(pdata, mdata);
sp_col = sp_precompute(sp, msh, 'gradient', false, 'divergence', false);
conn_e = sp_col.connectivity;
nel = msh.nel; nsh = sp_col.nsh_max; ndof = sp.ndof;

% Clamped at x = 0 face
ncp_dir = sp.scalar_spaces{1}.ndof_dir;
[i1, i2, i3] = ind2sub(ncp_dir, 1:sp.scalar_spaces{1}.ndof);
clamped_sc = find(i1 == 1);
ndof_sc = sp.scalar_spaces{1}.ndof;
clamped_dofs = [clamped_sc, clamped_sc + ndof_sc, clamped_sc + 2*ndof_sc];
free_dofs = setdiff(1:ndof, clamped_dofs);
free_mask = false(ndof, 1); free_mask(free_dofs) = true;

% External load: Center of tip face (x = L, y = h/2, z = w/2)
mid_y = round(ncp_dir(2)/2);
mid_z = round(ncp_dir(3)/2);
load_sc = find(i1 == ncp_dir(1) & abs(i2 - mid_y) <= 1 & abs(i3 - mid_z) <= 1);
F = zeros(ndof, 1);
load_dofs_y = load_sc + ndof_sc;
F(load_dofs_y) = -1.0 / numel(load_dofs_y);

% Template Element Precomputation
srf_tmpl = nrb4surf([0 0 0], [3*hx 0 0], [0 3*hy 0], [3*hx 3*hy 0]);
pdata_t.geo_name = nrbextrude(srf_tmpl, [0 0 3*hz]);
pdata_t.drchlt_sides = []; pdata_t.nmnn_sides = []; pdata_t.press_sides = []; pdata_t.symm_sides = [];
pdata_t.E = 1.0; pdata_t.nu = 0.3;
pdata_t.lambda_lame = @(x,y,z) 0.5769 * ones(size(x));
pdata_t.mu_lame     = @(x,y,z) 0.3846 * ones(size(x));
mdata_t.degree = [p, p, p]; mdata_t.regularity = [p-1, p-1, p-1];
mdata_t.nsub = [3 3 3]; mdata_t.nquad = [p+1, p+1, p+1];
[geo_t, msh_t, sp_t] = buildSpaces(pdata_t, mdata_t);
sp_col_t = sp_precompute(sp_t, msh_t, 'gradient', true, 'divergence', true);
msh_col_t = msh_precompute(msh_t);
l_val_t = 0.5769 * ones(msh_t.nqn, msh_t.nel);
m_val_t = 0.3846 * ones(msh_t.nqn, msh_t.nel);
[rt, ct, vt] = op_su_ev(sp_col_t, sp_col_t, msh_col_t, l_val_t, m_val_t);
ve_tmpl = single(reshape(vt, [nsh, nsh, 3, 3, 3]));

[ix, iy, iz] = ind2sub([nelx, nely, nelz], 1:nel);
tx = 2 * ones(1, nel); tx(ix == 1) = 1; tx(ix == nelx) = 3;
ty = 2 * ones(1, nel); ty(iy == 1) = 1; ty(iy == nely) = 3;
tz = 2 * ones(1, nel); tz(iz == 1) = 1; tz(iz == nelz) = 3;

vals_e = zeros(nsh, nsh, nel, 'single');
for e = 1:nel
    vals_e(:, :, e) = ve_tmpl(:, :, tx(e), ty(e), tz(e));
end

% Fast 3D Sensitivity Filter
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

% GPU Setup
conn_gpu = gpuArray(int32(conn_e));
vals_gpu = gpuArray(vals_e);
F_gpu = gpuArray(single(F));
free_mask_gpu = gpuArray(free_mask);
u_state = zeros(ndof, 1, 'like', F_gpu);

xPhys = volfrac * ones(nel, 1, 'single');
Emin = single(1e-3);
compliance_hist = zeros(max_iter, 1);
time_hist = zeros(max_iter, 1);

fprintf('Starting Optimization (%d iterations)...\n', max_iter);
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
    fprintf('Iter %2d: Compliance = %9.2f | Max Change = %.4f | Time = %.3f s\n', ...
        iter, c, max(abs(xPhys - x_old)), t_iter);
end

t_tot = toc(t_start);
fprintf('Completed in %.2f s (Avg: %.3f s/iter)\n', t_tot, mean(time_hist));

% Plot with correct coordinate mapping and camera angle
rho_3d = reshape(double(xPhys), [nelx, nely, nelz]);
% In meshgrid: dim1 is Y, dim2 is X, dim3 is Z
rho_perm = permute(rho_3d, [2, 1, 3]); % size [nely, nelx, nelz] = [24, 48, 24]
[X_mesh, Y_mesh, Z_mesh] = meshgrid(linspace(0, L, nelx), linspace(0, h, nely), linspace(0, w, nelz));

fig = figure('Color', 'w', 'Position', [100, 100, 1150, 480], 'Visible', 'off');
subplot(1, 2, 1);
rho_smooth = smooth3(rho_perm, 'box', 3);
p_iso = patch(isosurface(X_mesh, Y_mesh, Z_mesh, rho_smooth, 0.30));
isonormals(X_mesh, Y_mesh, Z_mesh, rho_smooth, p_iso);
set(p_iso, 'FaceColor', [0.15, 0.45, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.95);
view([-35, 25]); axis tight; axis equal;
camlight('headlight'); lighting gouraud;
box on; grid on;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k', 'FontSize', 10, 'LineWidth', 1.2);
xlabel('X (Length)', 'FontWeight', 'bold');
ylabel('Y (Height)', 'FontWeight', 'bold');
zlabel('Z (Width)', 'FontWeight', 'bold');
title('3D Optimal Space-Frame Cantilever (101,400 DOFs)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

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
title('Convergence History (101,400 DOFs)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Iteration', 'FontWeight', 'bold');

exportgraphics(fig, 'test_cantilever_center_load.png', 'Resolution', 300);
save('cantilever_center_results.mat', 'xPhys', 'compliance_hist', 'time_hist', 'nelx', 'nely', 'nelz');
fprintf('Saved test_cantilever_center_load.png and cantilever_center_results.mat\n');

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
