% DEMO_IMMERSED_GPU_SOLVE
% Solves a 3D linear elasticity problem directly on an immersed NIST STEP B-Rep
% using Catmull-Clark basis and FastFormation GPU Matrix-Free PCG solver.
clear; clc; close all;
repo_root = fullfile(fileparts(mfilename('fullpath')), '..');
addpath(genpath(fullfile(repo_root, 'src')));

step_file = fullfile(repo_root, 'Models', 'NIST-PMI-STEP-Files', 'NIST-PMI-STEP-Files', 'AP203 geometry only', 'nist_ctc_01_asme1_rd.stp');

fprintf('========================================================================\n');
fprintf('  IMMERSED IGA ON NIST STEP B-REP (FASTFORMATION GPU MATRIX-FREE)\n');
fprintf('========================================================================\n\n');

%% 1. Load NIST STEP CAD Model
brep = importBRep(step_file);

% Bounding box with 5% padding
minCoords = min(brep.nodes, [], 1);
maxCoords = max(brep.nodes, [], 1);
pad = 0.05 * (maxCoords - minCoords);
grid_bounds = [minCoords(1)-pad(1), maxCoords(1)+pad(1); ...
               minCoords(2)-pad(2), maxCoords(2)+pad(2); ...
               minCoords(3)-pad(3), maxCoords(3)+pad(3)];

% Background mesh resolution
grid_res = [24, 16, 12];
nelx = grid_res(1); nely = grid_res(2); nelz = grid_res(3);
nel = nelx * nely * nelz;

Lx = grid_bounds(1,2) - grid_bounds(1,1);
Ly = grid_bounds(2,2) - grid_bounds(2,1);
Lz = grid_bounds(3,2) - grid_bounds(3,1);
hx = Lx / nelx; hy = Ly / nely; hz = Lz / nelz;

%% 2. Background Spline Space & Geometry
p = 2;
sp = iga_space_box(grid_bounds, grid_res, p);
conn_e = sp.connectivity;
nsh = sp.nsh;
ndof = sp.ndof;

%% 3. Cut-Cell Quadrature & Immersed Integration Weights
fprintf('\nAssembling immersed element integration weights...\n');
elem_weights = assemble_immersed_element_weights(brep, grid_bounds, grid_res, [4, 4, 4]);

% Small background void modulus to prevent singular exterior DOFs
Emin = 1e-4;
scale = Emin + (1.0 - Emin) * elem_weights(:);

%% 4. FastFormation 3D Template Precomputation
t_tmpl = tic;
[Ke_types, type_id] = iga_elasticity_element_matrices(sp, 0.5769, 0.3846);
Ke_types = single(Ke_types);
vals_e = Ke_types(:, :, type_id);
fprintf('FastFormation Operator Ready in %.2f s (DOFs: %d)\n', toc(t_tmpl), ndof);

%% 5. Boundary Conditions & External Force
% Clamp DOFs at the bottom foundation face (Z = zmin)
ncp_dir = sp.ndof_dir;
[i1, i2, i3] = ind2sub(ncp_dir, 1:sp.ndof_sc);
clamped_sc = find(i3 == 1);
ndof_sc = sp.ndof_sc;
clamped_dofs = [clamped_sc, clamped_sc + ndof_sc, clamped_sc + 2*ndof_sc];
free_dofs = setdiff(1:ndof, clamped_dofs);
free_mask = false(ndof, 1); free_mask(free_dofs) = true;

% Apply downward force at top face center
mid_x = round(ncp_dir(1)/2);
mid_y = round(ncp_dir(2)/2);
load_sc = find(i3 == ncp_dir(3) & abs(i1 - mid_x) <= 2 & abs(i2 - mid_y) <= 2);
F = zeros(ndof, 1, 'single');
load_dofs_z = load_sc + 2*ndof_sc;
F(load_dofs_z) = -1.0 / numel(load_dofs_z);

%% 6. GPU Matrix-Free PCG State Solve
conn_gpu = gpuArray(int32(conn_e));
vals_gpu = gpuArray(vals_e);
F_gpu = gpuArray(F);
scale_gpu = gpuArray(single(scale));
free_mask_gpu = gpuArray(free_mask);

diag_vals = zeros(nsh, nel, 'like', vals_gpu);
for a = 1:nsh
    diag_vals(a, :) = squeeze(vals_gpu(a, a, :))' .* scale_gpu';
end
K_diag = accumarray(conn_gpu(:), diag_vals(:), [ndof, 1]);
M_inv = 1 ./ max(K_diag, single(1e-6));

matvec = @(p_in) eval_matvec_3d(p_in, conn_gpu, vals_gpu, scale_gpu, free_mask_gpu, ndof);
u_state = zeros(ndof, 1, 'single', 'gpuArray');
r = (F_gpu - matvec(u_state)) .* free_mask_gpu;
z = M_inv .* r;
p_vec = z;
rz_old = sum(r .* z);
tol = single(1e-4);
norm_f = norm(F_gpu(free_dofs));

fprintf('\nSolving Immersed Linear Elasticity on GPU via Matrix-Free PCG...\n');
t_pcg = tic;
for pcg_it = 1:250
    Ap = matvec(p_vec);
    pAp = sum(p_vec .* Ap);
    if abs(pAp) < 1e-12, break; end
    alpha = rz_old / pAp;
    u_state = u_state + alpha * p_vec;
    r = r - alpha * Ap;
    rel_res = norm(r) / norm_f;
    if rel_res < tol, break; end
    z = M_inv .* r;
    rz_new = sum(r .* z);
    p_vec = z + (rz_new / rz_old) * p_vec;
    rz_old = rz_new;
end
t_solve = toc(t_pcg);
u = gather(double(u_state));
compliance = gather(double(sum(F_gpu .* u_state)));

fprintf('PCG converged in %d iterations (Time: %.3f s, Compliance: %.4e)\n', ...
    pcg_it, t_solve, compliance);

%% 7. Visualize Displacement on Immersed B-Rep Surface
figDir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end

fig = figure('Color', 'w', 'Position', [100, 100, 1100, 500]);

% Left: Immersed Cut Elements
subplot(1, 2, 1);
patch('Faces', brep.elements, 'Vertices', brep.nodes, ...
      'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
hold on;
[Xg, Yg, Zg] = ndgrid(linspace(grid_bounds(1,1), grid_bounds(1,2), nelx), ...
                      linspace(grid_bounds(2,1), grid_bounds(2,2), nely), ...
                      linspace(grid_bounds(3,1), grid_bounds(3,2), nelz));
cut_mask = (elem_weights > 0.05 & elem_weights < 0.95);
scatter3(Xg(cut_mask), Yg(cut_mask), Zg(cut_mask), 35, elem_weights(cut_mask), 'filled');
colorbar; colormap(gca, 'parula');
axis equal tight; grid on; view(45, 30);
title('Immersed Cut-Cell Quadrature Weights w_e', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('X'); ylabel('Y'); zlabel('Z');

% Right: Displacement Field on B-Rep Surface
subplot(1, 2, 2);
% Interpolate displacement from background grid control points onto B-Rep nodes
ux = u(1:ndof_sc); uy = u(ndof_sc+1:2*ndof_sc); uz = u(2*ndof_sc+1:3*ndof_sc);
u_mag_cp = sqrt(ux.^2 + uy.^2 + uz.^2);
u_mag_grid = reshape(u_mag_cp, ncp_dir);

% Sample at B-Rep vertices
xg_vec = linspace(grid_bounds(1,1), grid_bounds(1,2), ncp_dir(1));
yg_vec = linspace(grid_bounds(2,1), grid_bounds(2,2), ncp_dir(2));
zg_vec = linspace(grid_bounds(3,1), grid_bounds(3,2), ncp_dir(3));

F_interp = griddedInterpolant({xg_vec, yg_vec, zg_vec}, u_mag_grid, 'linear', 'nearest');
u_on_brep = F_interp(brep.nodes(:,1), brep.nodes(:,2), brep.nodes(:,3));

patch('Faces', brep.elements, 'Vertices', brep.nodes, ...
      'FaceVertexCData', u_on_brep, 'FaceColor', 'interp', 'EdgeColor', [0.2 0.2 0.2], 'EdgeAlpha', 0.2);
colorbar; colormap(gca, 'jet');
axis equal tight; grid on; view(45, 30);
camlight('headlight'); lighting gouraud;
title('Displacement Magnitude |u| on NIST STEP Model', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('X'); ylabel('Y'); zlabel('Z');

figPath = fullfile(figDir, 'nist_step_immersed_gpu_solve.png');
exportgraphics(fig, figPath, 'Resolution', 150);
close(fig);
fprintf('Saved immersed solve figure: %s\n', figPath);

function y = eval_matvec_3d(p, conn, vals, scale, free_mask, ndof)
    p_act = p .* free_mask;
    pe = p_act(conn);
    [nsh_val, nel_val] = size(pe);
    pe_reshaped = reshape(pe, [nsh_val, 1, nel_val]);
    ye = pagemtimes(vals, pe_reshaped);
    ye_scaled = ye .* reshape(scale, [1, 1, nel_val]);
    y_full = accumarray(conn(:), ye_scaled(:), [ndof, 1]);
    y = y_full .* free_mask;
end
