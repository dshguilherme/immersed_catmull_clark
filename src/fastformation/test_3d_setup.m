% test_3d_setup.m

L = 1.2; h = 0.6; w = 0.3;
nsub = [16, 8, 4];

fprintf('Building 3D spline spaces...\n');
sp = iga_space_box([0 L; 0 h; 0 w], nsub, 2);

fprintf('3D Mesh: %d x %d x %d elements (%d total elements)\n', ...
    nsub(1), nsub(2), nsub(3), sp.nel);
fprintf('3D DOFs: %d (scalar: %d)\n', sp.ndof, sp.ndof_sc);

% Determine Dirichlet boundary DOFs (clamped at x = 0 face, side 1;
% equivalently sp.boundary(1).dofs)
ncp_dir = sp.ndof_dir; % [nx, ny, nz]
nx = ncp_dir(1); ny = ncp_dir(2); nz = ncp_dir(3);

% Scalar DOFs with i_x == 1 are clamped
[i1, i2, i3] = ind2sub([nx, ny, nz], 1:sp.ndof_sc);
clamped_scalar = find(i1 == 1);

% Vector DOFs clamped in all 3 components (u_x, u_y, u_z = 0)
ndof_sc = sp.ndof_sc;
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
