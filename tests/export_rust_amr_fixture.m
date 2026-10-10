function export_rust_amr_fixture()
% EXPORT_RUST_AMR_FIXTURE  MATLAB reference for the Rust AMR loop port.
%   Obstacle-3 L-bracket immersed in a [4 4 1] octree; clamp y = ymin, unit
%   downward load on x = xmax; 3 AMR cycles with the default indicator/marking.
%   Writes rust/tests/fixtures/amr_lbracket.json.

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', 'src')));
out = fullfile(here, '..', 'rust', 'tests', 'fixtures', 'amr_lbracket.json');

v = [0 0 0; 4 0 0; 4 2 0; 2 2 0; 2 4 0; 0 4 0; 0 0 1; 4 0 1; 4 2 1; 2 2 1; 2 4 1; 0 4 1];
f = [1 2 8; 1 8 7; 2 3 9; 2 9 8; 3 4 10; 3 10 9; 4 5 11; 4 11 10; 5 6 12; 5 12 11; 6 1 7; 6 7 12; ...
     1 4 2; 1 6 4; 4 6 5; 7 8 10; 7 10 12; 10 11 12];
brep.nodes = v; brep.elements = f;
gb = [-0.5 4.5; -0.5 4.5; -0.2 1.2];
oct = octree_mesh_3d(gb, [4 4 1], 0);
bc_fn = @(mesh) lbracket_bcs(mesh, gb);
[mesh, ~, vm, hist] = adaptive_mesh_refinement_loop(oct, brep, bc_fn, struct('max_amr_cycles', 3, 'E', 1e5, 'nu', 0.3));

data.nodes = v; data.elements = f - 1; data.grid_bounds = gb; data.grid_res = [4 4 1];
data.cycles = 3; data.E = 1e5; data.nu = 0.3;
data.n_elements = hist.n_elements(:).';
data.dofs = hist.dofs(:).';
data.compliance = hist.compliance(:).';
data.sigma_max = hist.sigma_max(:).';
data.final_levels = mesh.levels(:).';
data.final_vm = vm(:).';
fid = fopen(out, 'w'); fwrite(fid, jsonencode(data), 'char'); fclose(fid);
fprintf('Wrote %s (elements per cycle: %s)\n', out, mat2str(hist.n_elements(:).'));
end

function [fixed, F] = lbracket_bcs(mesh, gb)
mc = mesh.nodes(mesh.master_node_ids, :);
clamp = find(mc(:, 2) <= gb(2, 1) + 1e-6);
fixed = sort([(clamp - 1) * 3 + 1; (clamp - 1) * 3 + 2; (clamp - 1) * 3 + 3]).';
load = find(mc(:, 1) >= gb(1, 2) - 1e-6);
F = zeros(3 * mesh.n_master, 1);
F((load - 1) * 3 + 2) = -1 / numel(load);
end
