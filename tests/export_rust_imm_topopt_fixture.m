function export_rust_imm_topopt_fixture()
% EXPORT_RUST_IMM_TOPOPT_FIXTURE  MATLAB reference for the Rust immersed topology optimizer.
%   topopt_immersed_iga_3d on a tilted box (grid [7 6 5], 4 iterations, legacy
%   stabilization gamma = 0.05) with a tight single-precision PCG. Writes
%   rust/tests/fixtures/imm_topopt_tilted_box.json.

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', 'src')));
out = fullfile(here, '..', 'rust', 'tests', 'fixtures', 'imm_topopt_tilted_box.json');
v = [0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1] .* [1.6 0.8 0.6];
f = [1 2 6; 1 6 5; 2 3 7; 2 7 6; 3 4 8; 3 8 7; 4 1 5; 4 5 8; 1 4 3; 1 3 2; 5 6 7; 5 7 8];
a = deg2rad(17); b = deg2rad(9);
Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1]; Rx = [1 0 0; 0 cos(b) -sin(b); 0 sin(b) cos(b)];
brep.nodes = (Rx * Rz * v.').' + [0.13 0.07 0.05]; brep.elements = f;
set(0, 'DefaultFigureVisible', 'off');
res = topopt_immersed_iga_3d(brep, struct('grid_res', [7 6 5], 'max_iter', 4, 'pcg_tol', 1e-7, 'pcg_maxit', 20000, ...
    'output_fig', fullfile(tempdir, 'imm_topopt_fixture.png')));
close all force;
data.nodes = brep.nodes; data.elements = f - 1; data.grid_res = [7 6 5]; data.max_iter = 4;
data.volfrac = 0.35; data.penal = 3.0; data.rmin = 1.8; data.gamma_gp = 0.05;
data.compliance = res.compliance_hist(:).';
data.volume = res.vol_hist(:).';
data.xPhys = double(res.xPhys(:)).';
fid = fopen(out, 'w'); fwrite(fid, jsonencode(data), 'char'); fclose(fid);
fprintf('Wrote %s (compliance %s)\n', out, mat2str(res.compliance_hist(:).', 6));
end
