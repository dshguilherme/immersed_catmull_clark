function export_rust_contact_fixture()
% EXPORT_RUST_CONTACT_FIXTURE  MATLAB reference for the Rust multi-body contact port.
%   Obstacle-2 setup (punch on a block, 2x2x2 root grids): unilateral active-set
%   solve and bonded solve. Writes rust/tests/fixtures/contact_punch.json.

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', 'src')));
out = fullfile(here, '..', 'rust', 'tests', 'fixtures', 'contact_punch.json');

f = [1 2 6; 1 6 5; 2 3 7; 2 7 6; 3 4 8; 3 8 7; 4 1 5; 4 5 8; 1 4 3; 1 3 2; 5 6 7; 5 7 8];
base.nodes = [-5 -5 0; 5 -5 0; 5 5 0; -5 5 0; -5 -5 4; 5 -5 4; 5 5 4; -5 5 4]; base.elements = f;
punch.nodes = [-2 -2 4; 2 -2 4; 2 2 4; -2 2 4; -2 -2 7; 2 -2 7; 2 2 7; -2 2 7]; punch.elements = f;
b1.brep = base; b1.name = 'Base'; b1.grid_res = [2 2 2]; b1.max_level = 0;
b2.brep = punch; b2.name = 'Punch'; b2.grid_res = [2 2 2]; b2.max_level = 0;
asm = setup_assembly_3d({b1, b2}, struct('gap_tol', 0.5));
bc = {struct('body_idx', 1, 'type', 'dirichlet', 'filter', @(x,y,z) abs(z - 0) < 1e-3, 'value', [0 0 0], 'method', 'strong'); ...
      struct('body_idx', 2, 'type', 'neumann', 'filter', @(x,y,z) abs(z - 7) < 1e-3, 'value', [0 0 -25.0])};
[uU, rU] = solve_assembly_contact_3d(asm, zeros(asm.total_dof, 1), bc, struct('mode', 'unilateral', 'gamma_c', 50.0, 'max_iter', 10));
[uB, ~] = solve_assembly_contact_3d(asm, zeros(asm.total_dof, 1), bc, struct('mode', 'bonded', 'gamma_c', 50.0));

inter = asm.interfaces{1};
data.base_nodes = base.nodes; data.punch_nodes = punch.nodes; data.elements = f - 1;
data.gap_tol = 0.5; data.gamma_c = 50.0; data.E = 1e5; data.nu = 0.3;
data.total_dof = asm.total_dof;
data.ndof = [asm.bodies{1}.ndof, asm.bodies{2}.ndof];
data.n_pairs = inter.n_pairs;
data.pts_A = inter.pts_A; data.pts_B = inter.pts_B; data.normals = inter.normals; data.areas = inter.areas(:).';
data.facets_A = inter.facets_A(:).' - 1; data.facets_B = inter.facets_B(:).' - 1;
data.unilateral_u = uU.'; data.unilateral_iterations = rU.iterations; data.unilateral_converged = rU.converged;
data.unilateral_active = double(ismember(1:inter.n_pairs, rU.active_pairs));
data.unilateral_min_gap = rU.min_gap; data.unilateral_pressure = rU.contact_press(:).';
data.bonded_u = uB.';
fid = fopen(out, 'w'); fwrite(fid, jsonencode(data), 'char'); fclose(fid);
fprintf('Wrote %s (%d pairs, unilateral: %d iterations, %d active, min gap %.3e)\n', out, inter.n_pairs, rU.iterations, rU.n_active, rU.min_gap);
end
