% DEMO_OCTREE_IMMERSED_REFINEMENT
% Demonstrates adaptive 3D octree local refinement on the NIST STEP model
% using Catmull-Clark subdivision hierarchy for immersed IGA with high-contrast visuals.
clear; clc; close all;
this_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(this_dir, '..', 'src')));

step_file = fullfile(this_dir, '..', 'Models', 'NIST-PMI-STEP-Files', 'NIST-PMI-STEP-Files', 'AP203 geometry only', 'nist_ctc_01_asme1_rd.stp');

fprintf('========================================================================\n');
fprintf('  HIGH-CONTRAST ADAPTIVE 3D OCTREE REFINEMENT (CATMULL-CLARK IGA)\n');
fprintf('========================================================================\n\n');

% 1. Load NIST STEP CAD Model
brep = importBRep(step_file);

minCoords = min(brep.nodes, [], 1);
maxCoords = max(brep.nodes, [], 1);
pad = 0.05 * (maxCoords - minCoords);
grid_bounds = [minCoords(1)-pad(1), maxCoords(1)+pad(1); ...
               minCoords(2)-pad(2), maxCoords(2)+pad(2); ...
               minCoords(3)-pad(3), maxCoords(3)+pad(3)];

% Coarse background root grid
grid_res = [16, 12, 8];
nx = grid_res(1); ny = grid_res(2); nz = grid_res(3);
hx = (grid_bounds(1,2) - grid_bounds(1,1)) / nx;
hy = (grid_bounds(2,2) - grid_bounds(2,1)) / ny;
hz = (grid_bounds(3,2) - grid_bounds(3,1)) / nz;

% 2. Classify Root Cells to Identify Cut Boundary Region
fprintf('Classifying root background cells...\n');
status_root = classify_background_cells(brep, grid_bounds, grid_res);

% 3. Refine Cut Cells into Adaptive 2:1 Balanced Octree
refine_fn = @(cb) is_cell_cut_fast(cb, brep);

fprintf('\nConstructing adaptive octree mesh...\n');
octree = octree_mesh_3d(grid_bounds, grid_res, 1, refine_fn);

n_leaves = octree.n_leaves;
levels = octree.levels;
leaf_b = octree.bounds;

n_level0 = sum(levels == 0);
n_level1 = sum(levels == 1);
n_uniform = nx * ny * nz * 8; % Equivalent uniform fine mesh
savings_pct = (1.0 - n_leaves / n_uniform) * 100;

% 4. Multi-Panel High-Contrast Visualization
fig = figure('Color', 'w', 'Position', [60, 80, 1450, 480]);

%% Panel 1: 3D Boundary-Conformal Octree Refinement
subplot(1, 3, 1);
% 1. Solid-shaded CAD Model with metallic warm tint
patch('Faces', brep.elements, 'Vertices', brep.nodes, ...
      'FaceColor', [0.88 0.70 0.45], 'EdgeColor', [0.4 0.3 0.2], ...
      'EdgeAlpha', 0.15, 'FaceAlpha', 0.9);
hold on;

% 2. Draw refined cut cells (Level 1) as high-contrast blue translucent voxels
cut_leaf_indices = find(levels == 1);
% Render a representative subset of refined boundary cells for clarity
step_render = max(1, round(numel(cut_leaf_indices) / 350));
for idx = cut_leaf_indices(1:step_render:end)'
    cb = leaf_b(idx, :);
    plot_voxel_solid(cb, [0.15 0.55 0.95], 0.25, [0.05 0.35 0.85], 0.8);
end

% 3. Outer domain wireframe
plot_box_wireframe(grid_bounds(:)', [0.2 0.2 0.2], 1.0, '--');

axis equal tight; grid on; box on; view(45, 30);
xlim([grid_bounds(1,1)-20, grid_bounds(1,2)+20]);
ylim([grid_bounds(2,1)-20, grid_bounds(2,2)+20]);
zlim([grid_bounds(3,1)-20, grid_bounds(3,2)+20]);
camlight('headlight'); camlight('right'); lighting gouraud;
set(gca, 'Color', 'w', 'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2], 'ZColor', [0.2 0.2 0.2]);
xlabel('X', 'FontWeight', 'bold', 'Color', 'k');
ylabel('Y', 'FontWeight', 'bold', 'Color', 'k');
zlabel('Z', 'FontWeight', 'bold', 'Color', 'k');
title('3D Boundary-Conformal Octree', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');

%% Panel 2: 2D Mid-Plane Cross-Section (Z = Z_mid)
subplot(1, 3, 2);
z_mid = 0.5 * (minCoords(3) + maxCoords(3));

% Find leaf cells intersecting the Z_mid slice
slice_mask = (leaf_b(:, 5) <= z_mid & leaf_b(:, 6) >= z_mid);
slice_indices = find(slice_mask);

hold on;
% Draw coarse Level 0 cells first (light gray)
for idx = slice_indices'
    if levels(idx) == 0
        cb = leaf_b(idx, :);
        rectangle('Position', [cb(1), cb(3), cb(2)-cb(1), cb(4)-cb(3)], ...
                  'FaceColor', [0.94 0.95 0.97], 'EdgeColor', [0.65 0.65 0.70], ...
                  'LineWidth', 1.0);
    end
end

% Draw refined Level 1 cells (vivid cyan/blue fill with bold blue edges)
for idx = slice_indices'
    if levels(idx) == 1
        cb = leaf_b(idx, :);
        rectangle('Position', [cb(1), cb(3), cb(2)-cb(1), cb(4)-cb(3)], ...
                  'FaceColor', [0.35 0.75 1.00], 'EdgeColor', [0.05 0.35 0.80], ...
                  'LineWidth', 1.2);
    end
end

% Extract and plot the 2D CAD boundary intersection contour in bold crimson red
cad_segs = extract_brep_slice_contour(brep, z_mid);
if ~isempty(cad_segs)
    plot([cad_segs(:, 1), cad_segs(:, 3)]', [cad_segs(:, 2), cad_segs(:, 4)]', ...
         'Color', [0.85 0.05 0.05], 'LineWidth', 2.2);
end

axis equal tight; box on; grid on;
set(gca, 'Color', 'w', 'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2], ...
    'GridColor', [0.8 0.8 0.8], 'GridAlpha', 0.6);
xlim([grid_bounds(1,1), grid_bounds(1,2)]);
ylim([grid_bounds(2,1), grid_bounds(2,2)]);
xlabel('X', 'FontWeight', 'bold', 'Color', 'k');
ylabel('Y', 'FontWeight', 'bold', 'Color', 'k');
title(sprintf('Cross-Section at Z = %.1f', z_mid), 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');

% Add clean legend proxy handles with dark black text
h1 = plot(nan, nan, 's', 'MarkerFaceColor', [0.94 0.95 0.97], 'MarkerEdgeColor', [0.65 0.65 0.70], 'MarkerSize', 10);
h2 = plot(nan, nan, 's', 'MarkerFaceColor', [0.35 0.75 1.00], 'MarkerEdgeColor', [0.05 0.35 0.80], 'MarkerSize', 10);
h3 = plot(nan, nan, '-', 'Color', [0.85 0.05 0.05], 'LineWidth', 2.2);
lgd = legend([h1, h2, h3], {'Coarse Level 0', 'Refined Level 1', 'CAD Boundary \partial\Omega'}, ...
       'Location', 'northoutside', 'Orientation', 'horizontal', 'FontSize', 10, 'Box', 'off');
set(lgd, 'TextColor', [0.1 0.1 0.1]);

%% Panel 3: Computational Efficiency & Resolution Gain
subplot(1, 3, 3);
bar_vals = [nx*ny*nz, n_leaves, n_uniform];
bar_colors = [0.65 0.65 0.70; 0.20 0.55 0.90; 0.85 0.35 0.35];

b = bar(1:3, bar_vals, 0.55);
b.FaceColor = 'flat';
b.CData = bar_colors;
grid on; box on;
set(gca, 'Color', 'w', 'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2], ...
    'GridColor', [0.8 0.8 0.8], 'GridAlpha', 0.6, ...
    'XTick', 1:3, 'XTickLabel', {'Root Grid', 'Adaptive Octree', 'Uniform Fine'});
ylabel('Total Number of Elements', 'FontWeight', 'bold', 'Color', 'k');
title(sprintf('Element Efficiency (%.1f%% Savings)', savings_pct), ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');

for k = 1:3
    text(k, bar_vals(k) + max(bar_vals)*0.03, sprintf('%d', bar_vals(k)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', 'k');
end
ylim([0, max(bar_vals) * 1.18]);

% Export figure
figDir = fullfile(this_dir, '..', 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end
figPath = fullfile(figDir, 'fig_octree_nist_immersed_refinement.png');
exportgraphics(fig, figPath, 'Resolution', 200);
close(fig);

fprintf('\nHigh-contrast octree demonstration complete!\n');
fprintf('Exported figure: %s\n', figPath);

%% Helper Functions
function is_cut = is_cell_cut_fast(cb, brep)
    v1 = brep.nodes(brep.elements(:,1), :);
    v2 = brep.nodes(brep.elements(:,2), :);
    v3 = brep.nodes(brep.elements(:,3), :);
    tri_min = min(cat(3, v1, v2, v3), [], 3);
    tri_max = max(cat(3, v1, v2, v3), [], 3);
    
    overlaps = (tri_max(:,1) >= cb(1,1) & tri_min(:,1) <= cb(1,2) & ...
                tri_max(:,2) >= cb(2,1) & tri_min(:,2) <= cb(2,2) & ...
                tri_max(:,3) >= cb(3,1) & tri_min(:,3) <= cb(3,2));
    is_cut = any(overlaps);
end

function plot_box_wireframe(cb, color, alpha_val, line_style)
    if nargin < 4, line_style = '-'; end
    x = [cb(1) cb(2) cb(2) cb(1) cb(1) cb(1) cb(2) cb(2) cb(1) cb(1) cb(2) cb(2) cb(2) cb(2) cb(1) cb(1)];
    y = [cb(3) cb(3) cb(4) cb(4) cb(3) cb(3) cb(3) cb(4) cb(4) cb(3) cb(3) cb(3) cb(4) cb(4) cb(4) cb(4)];
    z = [cb(5) cb(5) cb(5) cb(5) cb(5) cb(6) cb(6) cb(6) cb(6) cb(6) cb(6) cb(5) cb(5) cb(6) cb(6) cb(5)];
    line(x, y, z, 'Color', [color, alpha_val], 'LineWidth', 1.0, 'LineStyle', line_style);
end

function plot_voxel_solid(cb, face_col, face_alp, edge_col, edge_alp)
    % 6 faces of a box
    verts = [cb(1) cb(3) cb(5); ...
             cb(2) cb(3) cb(5); ...
             cb(2) cb(4) cb(5); ...
             cb(1) cb(4) cb(5); ...
             cb(1) cb(3) cb(6); ...
             cb(2) cb(3) cb(6); ...
             cb(2) cb(4) cb(6); ...
             cb(1) cb(4) cb(6)];
    faces = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
    patch('Vertices', verts, 'Faces', faces, ...
          'FaceColor', face_col, 'FaceAlpha', face_alp, ...
          'EdgeColor', edge_col, 'EdgeAlpha', edge_alp, 'LineWidth', 0.8);
end

function segs = extract_brep_slice_contour(brep, z_s)
    v1 = brep.nodes(brep.elements(:, 1), :);
    v2 = brep.nodes(brep.elements(:, 2), :);
    v3 = brep.nodes(brep.elements(:, 3), :);
    
    z_min = min(cat(3, v1, v2, v3), [], 3);
    z_max = max(cat(3, v1, v2, v3), [], 3);
    
    cut_tris = find(z_min(:, 3) <= z_s & z_max(:, 3) >= z_s);
    segs = [];
    edges = [1 2; 2 3; 3 1];
    
    for idx = cut_tris'
        pts = [v1(idx, :); v2(idx, :); v3(idx, :)];
        x_pts = [];
        for e = 1:3
            pA = pts(edges(e, 1), :);
            pB = pts(edges(e, 2), :);
            if (pA(3) <= z_s && pB(3) >= z_s) || (pA(3) >= z_s && pB(3) <= z_s)
                if abs(pB(3) - pA(3)) > 1e-9
                    t = (z_s - pA(3)) / (pB(3) - pA(3));
                    P = pA + t * (pB - pA);
                    x_pts = [x_pts; P(1:2)];
                end
            end
        end
        if size(x_pts, 1) >= 2
            segs = [segs; x_pts(1, :), x_pts(2, :)];
        end
    end
end
