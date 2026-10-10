function export_rust_immersed_fixture()
% EXPORT_RUST_IMMERSED_FIXTURE  MATLAB reference for the Rust immersed solver.
%   Builds the same system as SOLVE_IMMERSED_IGA_3D (volume-fraction cut cells,
%   legacy stabilization, bottom clamp, default top load) for a tilted box B-Rep,
%   solves it in double precision with a direct solver, and writes
%   rust/tests/fixtures/immersed_tilted_box.json for rust/tests/immersed_vs_matlab.rs.

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', 'src')));
out = fullfile(here, '..', 'rust', 'tests', 'fixtures', 'immersed_tilted_box.json');

brep = tilted_box_brep();
grid_res = [7 6 5];
p = 2; E = 1; nu = 0.3; gamma_gp = 0.05; Emin = 1e-4;
lambda = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu));

minC = min(brep.nodes, [], 1); maxC = max(brep.nodes, [], 1);
pad = 0.05 * (maxC - minC);
grid_bounds = [minC(:) - pad(:), maxC(:) + pad(:)];
h_cell = (grid_bounds(:, 2) - grid_bounds(:, 1)).' ./ grid_res;

sp = iga_space_box(grid_bounds, grid_res, p);
[w, status] = assemble_immersed_element_weights(brep, grid_bounds, grid_res, [4 4 4]);
scale = Emin + (1 - Emin) * double(w(:));
[K_gp, ~] = assemble_ghost_penalty_stabilization(sp, sp.connectivity, status, grid_res, h_cell, gamma_gp);
[Ke, tid] = iga_elasticity_element_matrices(sp, lambda, mu);
[r, c] = iga_element_rows_cols(sp.connectivity);
K_vol = sparse(r, c, reshape(Ke(:, :, tid) .* reshape(scale, 1, 1, []), [], 1), sp.ndof, sp.ndof);
K = K_vol + K_gp;

% Boundary conditions exactly as solve_immersed_iga_3d (clamped 'bottom', default load)
[i1, i2, i3] = ind2sub(sp.ndof_dir, 1:sp.ndof_sc);
clamped_sc = find(i3 == 1);
free_mask = true(sp.ndof, 1);
free_mask([clamped_sc, clamped_sc + sp.ndof_sc, clamped_sc + 2 * sp.ndof_sc]) = false;
mid_x = round(sp.ndof_dir(1) / 2); mid_y = round(sp.ndof_dir(2) / 2);
load_sc = find(i3 == sp.ndof_dir(3) & abs(i1 - mid_x) <= 2 & abs(i2 - mid_y) <= 2);
F = zeros(sp.ndof, 1);
F(load_sc + 2 * sp.ndof_sc) = -1 / numel(load_sc);

u = zeros(sp.ndof, 1);
u(free_mask) = K(free_mask, free_mask) \ F(free_mask);

i = (1:sp.ndof).';
data.nodes = brep.nodes;
data.elements = brep.elements - 1;            % 0-based
data.grid_res = grid_res;
data.degree = p; data.young = E; data.poisson = nu; data.gamma_gp = gamma_gp; data.emin = Emin;
data.grid_bounds = grid_bounds;
data.status = reshape(status, 1, []);         % element-major, direction 0 fastest
data.weights = reshape(double(w), 1, []);
data.kgp_frobenius = norm(K_gp, 'fro');
data.kgp_probe = (K_gp * sin(0.37 * i)).';
data.k_probe = (K * sin(0.37 * i)).';
data.load = F.';
data.free_mask = double(free_mask.');
data.u_direct = u.';
data.compliance = F.' * u;
fid = fopen(out, 'w');
fwrite(fid, jsonencode(data), 'char');
fclose(fid);
fprintf('Wrote %s (ndof = %d, compliance = %.10e)\n', out, sp.ndof, data.compliance);
end

function brep = tilted_box_brep()
% Box [0,1.6]x[0,0.8]x[0,0.6], rotated 17 deg about z and 9 deg about x, shifted.
v = [0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1] .* [1.6 0.8 0.6];
f = [1 2 6; 1 6 5; 2 3 7; 2 7 6; 3 4 8; 3 8 7; 4 1 5; 4 5 8; 1 4 3; 1 3 2; 5 6 7; 5 7 8];
a = deg2rad(17); b = deg2rad(9);
Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];
Rx = [1 0 0; 0 cos(b) -sin(b); 0 sin(b) cos(b)];
brep.nodes = (Rx * Rz * v.').' + [0.13 0.07 0.05];
brep.elements = f;
end
