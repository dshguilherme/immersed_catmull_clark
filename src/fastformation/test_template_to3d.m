% TEST_TEMPLATE_TO3D
% Verify the in-house exact element precomputation against GeoPDEs op_su_ev
% (optional reference; set GEOPDES_PATH). The regression test
% tests/testIgaKernel.m covers the same check without GeoPDEs.
clear; clc;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'benchmarks'));

nelx = 16; nely = 8; nelz = 4;
L = 1.2; h = 0.6; w = 0.3;

% 1. In-house exact element matrices (one per distinct element type)
t0 = tic;
sp = iga_space_box([0 L; 0 h; 0 w], [nelx, nely, nelz], 2);
[Ke_types, type_id] = iga_elasticity_element_matrices(sp, 0.5769, 0.3846);
vals_fast = Ke_types(:, :, type_id);
t_fast = toc(t0);
fprintf('In-house precomputation took: %.4f s (%d element types)\n', t_fast, size(Ke_types, 3));

if ~geopdes_baseline_available()
    fprintf('GeoPDEs not available (set GEOPDES_PATH): skipping the external comparison.\n');
    return;
end

% 2. Standard GeoPDEs precomputation
pdata.geo_name = nrbextrude(nrb4surf([0 0 0], [L 0 0], [0 h 0], [L h 0]), [0 0 w]);
pdata.drchlt_sides = []; pdata.nmnn_sides = []; pdata.press_sides = []; pdata.symm_sides = [];
mdata.degree = [2 2 2]; mdata.regularity = [1 1 1];
mdata.nsub = [nelx, nely, nelz]; mdata.nquad = [3 3 3];
[~, msh, spg] = buildSpaces(pdata, mdata);
sp_col = sp_precompute(spg, msh, 'gradient', true, 'divergence', true);
msh_col = msh_precompute(msh);
t0 = tic;
[~, ~, v_std] = op_su_ev(sp_col, sp_col, msh_col, 0.5769 * ones(msh.nqn, msh.nel), 0.3846 * ones(msh.nqn, msh.nel));
fprintf('Standard op_su_ev took: %.4f s\n', toc(t0));

ve_std = reshape(v_std, [sp.nsh, sp.nsh, sp.nel]);
diff_max = max(abs(ve_std(:) - vals_fast(:)));
fprintf('Max difference between GeoPDEs and in-house: %e\n', diff_max);
assert(diff_max < 1e-12, 'Element matrices do not match!');
fprintf('SUCCESS: in-house element matrices match GeoPDEs.\n');
