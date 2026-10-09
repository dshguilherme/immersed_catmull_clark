% test_3d_setup.m
addpath(genpath('C:\Users\dshgu\OneDrive\Documents\geopdes-master'));
addpath('c:\Users\dshgu\OneDrive\Documents\FastFormation');

L = 1.2; h = 0.6; w = 0.3;
srf = nrb4surf([0 0 0], [L 0 0], [0 h 0], [L h 0]);
vol = nrbextrude(srf, [0 0 w]);

problem_data.geo_name = vol;
problem_data.drchlt_sides = [];
problem_data.nmnn_sides   = [];
problem_data.press_sides  = [];
problem_data.symm_sides   = [];
problem_data.E  = 1.0;
problem_data.nu = 0.3;
problem_data.lambda_lame = @(x, y, z) (0.3*1.0/((1+0.3)*(1-2*0.3)) * ones(size(x)));
problem_data.mu_lame     = @(x, y, z) (1.0/(2*(1+0.3)) * ones(size(x)));

method_data.degree     = [2, 2, 2];
method_data.regularity = [1, 1, 1];
method_data.nsub       = [16, 8, 4];
method_data.nquad      = [3, 3, 3];

fprintf('Building 3D spline spaces...\n');
[geometry, msh, sp] = buildSpaces(problem_data, method_data);

fprintf('3D Mesh: %d x %d x %d elements (%d total elements)\n', ...
    method_data.nsub(1), method_data.nsub(2), method_data.nsub(3), msh.nel);
fprintf('3D DOFs: %d (scalar: %d)\n', sp.ndof, sp.scalar_spaces{1}.ndof);

% Determine Dirichlet boundary DOFs (clamped at x = 0 face, side 1)
% In GeoPDEs 3D, boundary faces can be obtained from geometry / space
% Let's find DOFs at x = 0
ncp_dir = sp.scalar_spaces{1}.ndof_dir; % [nx, ny, nz]
nx = ncp_dir(1); ny = ncp_dir(2); nz = ncp_dir(3);

% Scalar DOFs with i_x == 1 are clamped
[i1, i2, i3] = ind2sub([nx, ny, nz], 1:sp.scalar_spaces{1}.ndof);
clamped_scalar = find(i1 == 1);

% Vector DOFs clamped in all 3 components (u_x, u_y, u_z = 0)
ndof_sc = sp.scalar_spaces{1}.ndof;
clamped_dofs = [clamped_scalar, clamped_scalar + ndof_sc, clamped_scalar + 2*ndof_sc];
free_dofs = setdiff(1:sp.ndof, clamped_dofs);
fprintf('Clamped DOFs: %d, Free DOFs: %d\n', numel(clamped_dofs), numel(free_dofs));

% Apply vertical downward load at the center of the free face (x = L)
% Center of face x = L is i1 == nx, i2 around ny/2, i3 around nz/2
load_sc = find(i1 == nx & abs(i2 - ny/2) <= 1 & abs(i3 - nz/2) <= 1);
F = zeros(sp.ndof, 1);
% Component 2 (y direction downwards) or Component 3 (z direction)
% Downward vertical is y direction:
load_dofs_y = load_sc + ndof_sc;
F(load_dofs_y) = -1.0 / numel(load_dofs_y);
fprintf('External load applied on %d DOFs at x = L (Total load: %.2f)\n', ...
    numel(load_dofs_y), sum(F(load_dofs_y)));

fprintf('3D Setup successfully verified!\n');
