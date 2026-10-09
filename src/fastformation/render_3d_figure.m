% RENDER_3D_FIGURE
% Render publication-quality 3D isosurface plot for 101,184 DOF optimization
clear; clc; close all;

load('topopt_3d_100k_results.mat');

L = 1.2; h = 0.6; w = 0.3;
max_iter = numel(compliance_hist);

fig = figure('Color', 'w', 'Position', [100, 100, 1150, 480], 'Visible', 'off');

% Left: 3D Optimal Structural Topology
subplot(1, 2, 1);
rho_3d = reshape(double(xPhys), [nelx, nely, nelz]);
rho_smooth = smooth3(rho_3d, 'box', 3);

% Physical coordinate grid matching dimensions [0, L] x [0, h] x [0, w]
[X, Y, Z] = meshgrid(linspace(0, h, nely), linspace(0, L, nelx), linspace(0, w, nelz));
% Note: In MATLAB meshgrid, dim 1 is y (rows), dim 2 is x (cols). 
% Permute so X matches L, Y matches h, Z matches w:
[X_mesh, Y_mesh, Z_mesh] = meshgrid(linspace(0, L, nelx), linspace(0, h, nely), linspace(0, w, nelz));
rho_perm = permute(rho_smooth, [2, 1, 3]);

p_iso = patch(isosurface(X_mesh, Y_mesh, Z_mesh, rho_perm, 0.35));
isonormals(X_mesh, Y_mesh, Z_mesh, rho_perm, p_iso);

set(p_iso, 'FaceColor', [0.15, 0.45, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.95);
view(3); axis tight; axis equal;
camlight('headlight'); lighting gouraud;
box on; grid on;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k', 'FontSize', 10, 'LineWidth', 1.2);
xlabel('X (Length)', 'FontWeight', 'bold');
ylabel('Y (Height)', 'FontWeight', 'bold');
zlabel('Z (Width)', 'FontWeight', 'bold');
title(sprintf('3D Optimal Topology (%dx%dx%d, 101,184 DOFs, GPU Matrix-Free FP32)', nelx, nely, nelz), ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

% Right: Convergence History
subplot(1, 2, 2);
yyaxis left;
plot(1:max_iter, compliance_hist, 'b-o', 'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'b');
ylabel('Compliance J = F^T U', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'b');
set(gca, 'YColor', 'b');

yyaxis right;
semilogy(1:max_iter, abs(diff([compliance_hist(1); compliance_hist])), 'r--s', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'r');
ylabel('Step Change | \Delta J |', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r');
set(gca, 'YColor', 'r');

grid on;
set(gca, 'FontSize', 10, 'XColor', 'k', 'Color', 'w', 'Box', 'on', 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
title('3D Convergence History (101,184 DOFs)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Iteration', 'FontWeight', 'bold');

fig_name = 'fig_topopt_3d_cantilever.png';
exportgraphics(fig, fig_name, 'Resolution', 300);
fprintf('Successfully generated %s for 101,184 DOFs!\n', fig_name);
