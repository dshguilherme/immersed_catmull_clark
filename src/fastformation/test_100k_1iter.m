% TEST_100K_1ITER
% Test 1 iteration of 3D topology optimization on 101,184 DOFs with GPU Matrix-Free
clear; clc;
addpath(genpath('C:\Users\dshgu\OneDrive\Documents\geopdes-master'));
addpath('C:\Users\dshgu\OneDrive\Documents\FastFormation');

nelx = 60; nely = 30; nelz = 15;
L = 1.2; h = 0.6; w = 0.3;
volfrac = 0.3; penal = 3.0; rmin = 1.5;

fprintf('1. Building 3D Spaces (nel: %dx%dx%d)...\n', nelx, nely, nelz);
hx = L / nelx; hy = h / nely; hz = w / nelz;
srf = nrb4surf([0 0 0], [L 0 0], [0 h 0], [L h 0]);
vol = nrbextrude(srf, [0 0 w]);

pdata.geo_name = vol;
pdata.drchlt_sides = []; pdata.nmnn_sides = []; pdata.press_sides = []; pdata.symm_sides = [];
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
nel = msh.nel;
nsh = sp_col.nsh_max;
ndof = sp.ndof;
fprintf('Mesh: %d elements, %d DOFs\n', nel, ndof);

% Boundary conditions: clamped at x = 0
ncp_dir = sp.scalar_spaces{1}.ndof_dir;
[i1, i2, i3] = ind2sub(ncp_dir, 1:sp.scalar_spaces{1}.ndof);
clamped_sc = find(i1 == 1);
ndof_sc = sp.scalar_spaces{1}.ndof;
clamped_dofs = [clamped_sc, clamped_sc + ndof_sc, clamped_sc + 2*ndof_sc];
free_dofs = setdiff(1:ndof, clamped_dofs);
free_mask = false(ndof, 1); free_mask(free_dofs) = true;

% External load: Downward at bottom center of x = L face
load_sc = find(i1 == ncp_dir(1) & abs(i2 - 1) <= 1 & abs(i3 - ncp_dir(3)/2) <= 1);
F = zeros(ndof, 1);
load_dofs_y = load_sc + ndof_sc;
F(load_dofs_y) = -1.0 / numel(load_dofs_y);

fprintf('2. Fast Template Element Precomputation...\n');
t0 = tic;
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

% Pre-populate vals_e directly in single precision
vals_e = zeros(nsh, nsh, nel, 'single');
for e = 1:nel
    vals_e(:, :, e) = ve_tmpl(:, :, tx(e), ty(e), tz(e));
end
fprintf('Template mapping done in: %.2f s\n', toc(t0));

fprintf('3. Fast 3D Sensitivity Filter Precomputation...\n');
t0 = tic;
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
fprintf('Filter done in: %.2f s\n', toc(t0));

fprintf('4. Transferring to GPU (RTX 2050)...\n');
t0 = tic;
conn_gpu = gpuArray(int32(conn_e));
vals_gpu = gpuArray(vals_e);
F_gpu = gpuArray(single(F));
free_mask_gpu = gpuArray(free_mask);
u_state = zeros(ndof, 1, 'like', F_gpu);
fprintf('GPU transfer done in: %.2f s\n', toc(t0));

fprintf('5. Benchmarking 1 PCG solve with 101,184 DOFs on GPU...\n');
xPhys = volfrac * ones(nel, 1, 'single');
Emin = single(1e-3);
scale = Emin + (1 - Emin) * (xPhys.^penal);
scale_gpu = gpuArray(scale);

% Diagonal Preconditioner on GPU
t_pcg = tic;
diag_vals = zeros(nsh, nel, 'like', vals_gpu);
for a = 1:nsh
    diag_vals(a, :) = squeeze(vals_gpu(a, a, :))' .* scale_gpu';
end
K_diag = accumarray(conn_gpu(:), diag_vals(:), [ndof, 1]);
M_inv = 1 ./ max(K_diag, single(1e-6));

matvec = @(p) eval_matvec_3d(p, conn_gpu, vals_gpu, scale_gpu, free_mask_gpu, ndof);
r = (F_gpu - matvec(u_state)) .* free_mask_gpu;
z = M_inv .* r;
p_vec = z;
rz_old = sum(r .* z);
tol = single(1e-4);
norm_f = norm(F_gpu(free_dofs));

for pcg_it = 1:100
    Ap = matvec(p_vec);
    pAp = sum(p_vec .* Ap);
    if abs(pAp) < 1e-12, break; end
    alpha = rz_old / pAp;
    u_state = u_state + alpha * p_vec;
    r = r - alpha * Ap;
    rel_res = norm(r) / norm_f;
    if rel_res < tol, break; end
    z = M_inv .* r;
    rz_new = sum(r .* z);
    p_vec = z + (rz_new / rz_old) * p_vec;
    rz_old = rz_new;
end
t_pcg_total = toc(t_pcg);
fprintf('PCG converged in %d iters, time: %.3f s (%.2f ms/iter) | rel_res: %.2e\n', ...
    pcg_it, t_pcg_total, (t_pcg_total/pcg_it)*1000, rel_res);

% Sensitivity on GPU
t_sens = tic;
ue = u_state(conn_gpu);
ke_ue = pagemtimes(vals_gpu, reshape(ue, [nsh, 1, nel]));
Ee = sum(ue .* squeeze(ke_ue), 1)';
c = gather(double(sum(F_gpu .* u_state)));
t_sens_time = toc(t_sens);
fprintf('Sensitivity evaluation: %.3f s | Compliance J = %.4f\n', t_sens_time, c);
fprintf('TOTAL 1-ITERATION TIME: %.3f s\n', t_pcg_total + t_sens_time);

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
