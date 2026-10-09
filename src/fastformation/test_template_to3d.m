% TEST_TEMPLATE_TO3D
% Verify template-based 3D element precomputation against standard op_su_ev
clear; clc;
addpath(genpath('C:\Users\dshgu\OneDrive\Documents\geopdes-master'));
addpath('C:\Users\dshgu\OneDrive\Documents\FastFormation');

nelx = 16; nely = 8; nelz = 4;
L = 1.2; h = 0.6; w = 0.3;
hx = L / nelx; hy = h / nely; hz = w / nelz;

% 1. Standard GeoPDEs precomputation
pdata.geo_name = nrbextrude(nrb4surf([0 0 0], [L 0 0], [0 h 0], [L h 0]), [0 0 w]);
pdata.drchlt_sides = []; pdata.nmnn_sides = []; pdata.press_sides = []; pdata.symm_sides = [];
pdata.E = 1.0; pdata.nu = 0.3;
pdata.lambda_lame = @(x,y,z) 0.5769 * ones(size(x));
pdata.mu_lame     = @(x,y,z) 0.3846 * ones(size(x));
mdata.degree = [2 2 2]; mdata.regularity = [1 1 1];
mdata.nsub = [nelx, nely, nelz]; mdata.nquad = [3 3 3];

fprintf('Building full spaces (%dx%dx%d)...\n', nelx, nely, nelz);
[geo, msh, sp] = buildSpaces(pdata, mdata);
sp_col = sp_precompute(sp, msh, 'gradient', true, 'divergence', true);
msh_col = msh_precompute(msh);
l_val = 0.5769 * ones(msh.nqn, msh.nel);
m_val = 0.3846 * ones(msh.nqn, msh.nel);
t0 = tic;
[r_std, c_std, v_std] = op_su_ev(sp_col, sp_col, msh_col, l_val, m_val);
t_std = toc(t0);
fprintf('Standard op_su_ev took: %.4f s\n', t_std);

% 2. Fast Template Precomputation
t0 = tic;
srf_tmpl = nrb4surf([0 0 0], [3*hx 0 0], [0 3*hy 0], [3*hx 3*hy 0]);
pdata_t.geo_name = nrbextrude(srf_tmpl, [0 0 3*hz]);
pdata_t.drchlt_sides = []; pdata_t.nmnn_sides = []; pdata_t.press_sides = []; pdata_t.symm_sides = [];
pdata_t.E = 1.0; pdata_t.nu = 0.3;
pdata_t.lambda_lame = @(x,y,z) 0.5769 * ones(size(x));
pdata_t.mu_lame     = @(x,y,z) 0.3846 * ones(size(x));
mdata_t.degree = [2 2 2]; mdata_t.regularity = [1 1 1];
mdata_t.nsub = [3 3 3]; mdata_t.nquad = [3 3 3];
[geo_t, msh_t, sp_t] = buildSpaces(pdata_t, mdata_t);
sp_col_t = sp_precompute(sp_t, msh_t, 'gradient', true, 'divergence', true);
msh_col_t = msh_precompute(msh_t);
l_val_t = 0.5769 * ones(msh_t.nqn, msh_t.nel);
m_val_t = 0.3846 * ones(msh_t.nqn, msh_t.nel);
[rt, ct, vt] = op_su_ev(sp_col_t, sp_col_t, msh_col_t, l_val_t, m_val_t);
nsh = sp_col.nsh_max;
ve_tmpl = reshape(vt, [nsh, nsh, 3, 3, 3]);

% Map templates to full elements
[ix, iy, iz] = ind2sub([nelx, nely, nelz], 1:msh.nel);
tx = 2 * ones(1, msh.nel); tx(ix == 1) = 1; tx(ix == nelx) = 3;
ty = 2 * ones(1, msh.nel); ty(iy == 1) = 1; ty(iy == nely) = 3;
tz = 2 * ones(1, msh.nel); tz(iz == 1) = 1; tz(iz == nelz) = 3;

vals_fast = zeros(nsh, nsh, msh.nel);
for e = 1:msh.nel
    vals_fast(:, :, e) = ve_tmpl(:, :, tx(e), ty(e), tz(e));
end
t_fast = toc(t0);
fprintf('Template precomputation took: %.4f s\n', t_fast);

ve_std = reshape(v_std, [nsh, nsh, msh.nel]);
diff_max = max(abs(ve_std(:) - vals_fast(:)));
fprintf('Max difference between standard and template: %e\n', diff_max);
assert(diff_max < 1e-12, 'Template matching failed!');
fprintf('SUCCESS: Template method is 100%% exact!\n');
