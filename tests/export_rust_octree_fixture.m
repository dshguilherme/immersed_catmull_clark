function export_rust_octree_fixture()
% EXPORT_RUST_OCTREE_FIXTURE  MATLAB reference for the Rust octree / MPC stack.
%   Two cases mirroring the obstacle course: (A) refined + balanced octree on a box
%   without immersed geometry (patch-test mesh), (B) the obstacle-5 mesh with a box
%   B-Rep (cut-cell weights), strong Dirichlet and Neumann BCs. Writes
%   rust/tests/fixtures/octree_cases.json for rust/tests/octree_vs_matlab.rs.

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', 'src')));
out = fullfile(here, '..', 'rust', 'tests', 'fixtures', 'octree_cases.json');

nodes_box = [0 0 0; 2 0 0; 2 2 0; 0 2 0; 0 0 2; 2 0 2; 2 2 2; 0 2 2];
elems_box = [1 2 6; 1 6 5; 2 3 7; 2 7 6; 3 4 8; 3 8 7; 4 1 5; 4 5 8; 1 4 3; 1 3 2; 5 6 7; 5 7 8];
brep_box.nodes = nodes_box; brep_box.elements = elems_box;

% Case A: obstacle-1 mesh
oct = octree_mesh_3d([0 2; 0 2; 0 2], [2 2 2], 1);
oct = subdivide_octree_leaves(oct, [1 4]);
oct = balance_octree_3d(oct);
mA = octree_structural_mesh(oct, [], struct('E', 1e5, 'nu', 0.25));
A = mesh_record(mA);
A.grid_bounds = [0 2; 0 2; 0 2]; A.grid_res = [2 2 2]; A.max_level = 1; A.subdivide = [1 4] - 1;
A.E = 1e5; A.nu = 0.25; A.has_brep = 0;

% Case B: obstacle-5 mesh with immersed box, strong Dirichlet + Neumann
oct = octree_mesh_3d([-1 11; -1 11; -1 11], [3 3 3], 1);
oct = subdivide_octree_leaves(oct, [1 2 5 10]);
oct = balance_octree_3d(oct);
mB = octree_structural_mesh(oct, brep_box, struct('E', 1e5, 'nu', 0.25));
B = mesh_record(mB);
B.grid_bounds = [-1 11; -1 11; -1 11]; B.grid_res = [3 3 3]; B.max_level = 1; B.subdivide = [1 2 5 10] - 1;
B.E = 1e5; B.nu = 0.25; B.has_brep = 1;
bc = {struct('type', 'dirichlet', 'filter', @(x,y,z) abs(z - (-1)) < 2.0, 'value', [0 0 0], 'method', 'strong'); ...
      struct('type', 'neumann', 'filter', @(x,y,z) abs(z - 11) < 2.0, 'value', [10.0 0 0])};
[~, F, st] = apply_boundary_conditions(mB, brep_box, bc);
B.force = F.';
B.fixed_dofs = st.fixed_dofs(:).' - 1;
u = zeros(3 * mB.n_master, 1);
if ~isempty(st.free_dofs) && norm(F) > 0
    u(st.free_dofs) = mB.K_master(st.free_dofs, st.free_dofs) \ F(st.free_dofs);
end
B.u_direct = u.';
% Case B with Neumann pressure on the top face and Robin springs on x = 0 (no solve)
bc2 = {struct('type', 'neumann', 'filter', @(x,y,z) abs(z - 2) < 1e-6, 'value', 3.0); ...
       struct('type', 'robin', 'filter', @(x,y,z) abs(x) < 1e-6, 'k_spring', 50.0)};
[Kb2, F2, ~] = apply_boundary_conditions(mB, brep_box, bc2);
i = (1:3 * mB.n_master).';
B.bc2_force = F2.';
B.bc2_k_probe = (Kb2 * sin(0.37 * i)).';

data.brep_nodes = nodes_box; data.brep_elements = elems_box - 1;
data.cases = {A, B};
fid = fopen(out, 'w'); fwrite(fid, jsonencode(data), 'char'); fclose(fid);
fprintf('Wrote %s (A: %d master nodes, %d hanging; B: %d master, |F| = %.3e)\n', out, mA.n_master, sum(mA.is_hanging), mB.n_master, norm(F));
end

function r = mesh_record(m)
r.n_nodes = m.n_nodes;
r.n_master = m.n_master;
r.n_hanging = sum(m.is_hanging);
r.n_elements = m.n_elements;
r.levels = m.levels(:).';
r.leaf_bounds = m.bounds;                     % [n_leaves x 6]
r.nodes = m.nodes;
r.elem_nodes = m.elem_nodes - 1;
r.master_ids = m.master_node_ids(:).' - 1;
r.weights = m.weights(:).';
i = (1:3 * m.n_master).';
r.k_probe = (m.K_master * sin(0.37 * i)).';
r.k_frobenius = norm(m.K_master, 'fro');
end
