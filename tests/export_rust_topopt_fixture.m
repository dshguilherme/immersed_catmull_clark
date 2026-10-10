function export_rust_topopt_fixture()
% EXPORT_RUST_TOPOPT_FIXTURE  MATLAB reference for the Rust 3D topology optimizer.
%   Runs topopt_iga_3d (CPU direct solver) on an 8x4x2 cantilever for 6 iterations
%   and writes rust/tests/fixtures/topopt3d_8x4x2.json (compliance history and final
%   densities) for rust/tests/topopt_vs_matlab.rs.

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', 'src')));
out = fullfile(here, '..', 'rust', 'tests', 'fixtures', 'topopt3d_8x4x2.json');
old = cd(tempdir);
cleanup = onCleanup(@() cd(old));
set(0, 'DefaultFigureVisible', 'off');
[xPhys, c_hist] = topopt_iga_3d(8, 4, 2, 0.3, 3.0, 1.5, 6, 'cpu');
close all force;
data.nel = [8 4 2]; data.volfrac = 0.3; data.penal = 3.0; data.rmin = 1.5; data.max_iter = 6;
data.compliance = c_hist(:).';
data.xPhys = xPhys(:).';
fid = fopen(out, 'w');
fwrite(fid, jsonencode(data), 'char');
fclose(fid);
fprintf('Wrote %s\n', out);
end
