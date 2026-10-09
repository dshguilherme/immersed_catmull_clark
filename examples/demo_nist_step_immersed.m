% DEMO_NIST_STEP_IMMERSED
% Imports a NIST STEP benchmark model and classifies an embedding Catmull-Clark
% background grid into interior, exterior, and cut elements.
clear; clc; close all;
addpath(genpath('src'));

step_file = 'C:/Users/dshgu/immersed-iga/Models/NIST-PMI-STEP-Files/NIST-PMI-STEP-Files/AP203 geometry only/nist_ctc_01_asme1_rd.stp';

fprintf('Loading NIST STEP CAD Model...\n');
brep = importBRep(step_file);

% Get bounding box with slight padding for immersed background grid
minCoords = min(brep.nodes, [], 1);
maxCoords = max(brep.nodes, [], 1);
pad = 0.05 * (maxCoords - minCoords);
grid_bounds = [minCoords(1)-pad(1), maxCoords(1)+pad(1); ...
               minCoords(2)-pad(2), maxCoords(2)+pad(2); ...
               minCoords(3)-pad(3), maxCoords(3)+pad(3)];

% 3D background grid resolution
grid_res = [24, 16, 12];
fprintf('Classifying 3D background cells (%dx%dx%d = %d cells)...\n', ...
    grid_res(1), grid_res(2), grid_res(3), prod(grid_res));

status = classify_background_cells(brep, grid_bounds, grid_res);

% Visualizing the immersed B-Rep and cut elements
fig = figure('Color', 'w', 'Position', [100, 100, 1100, 500]);

% Left: CAD Surface
subplot(1, 2, 1);
patch('Faces', brep.elements, 'Vertices', brep.nodes, ...
      'FaceColor', [0.2 0.6 0.9], 'EdgeColor', [0.1 0.3 0.6], 'FaceAlpha', 0.85);
axis equal tight; grid on; view(45, 30);
camlight('headlight'); lighting gouraud;
title('NIST CTC-01 STEP B-Rep Model', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('X'); ylabel('Y'); zlabel('Z');

% Right: Cut Cell Overlay
subplot(1, 2, 2);
% Plot semi-transparent B-Rep
patch('Faces', brep.elements, 'Vertices', brep.nodes, ...
      'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
hold on;

% Plot centroids of cut elements
[Xg, Yg, Zg] = ndgrid(linspace(grid_bounds(1,1), grid_bounds(1,2), grid_res(1)), ...
                      linspace(grid_bounds(2,1), grid_bounds(2,2), grid_res(2)), ...
                      linspace(grid_bounds(3,1), grid_bounds(3,2), grid_res(3)));
cut_pts = [Xg(status == -1), Yg(status == -1), Zg(status == -1)];
in_pts  = [Xg(status == 1),  Yg(status == 1),  Zg(status == 1)];

plot3(in_pts(:,1), in_pts(:,2), in_pts(:,3), 'g.', 'MarkerSize', 8);
plot3(cut_pts(:,1), cut_pts(:,2), cut_pts(:,3), 'r.', 'MarkerSize', 10);
hold off;
axis equal tight; grid on; view(45, 30);
legend({'B-Rep Surface', 'Interior Cells', 'Cut Cells'}, 'Location', 'northeast');
title('Immersed Catmull-Clark Cell Classification', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('X'); ylabel('Y'); zlabel('Z');

figDir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end
figPath = fullfile(figDir, 'nist_step_immersed_classification.png');
exportgraphics(fig, figPath, 'Resolution', 150);
close(fig);
fprintf('Saved classification visualization to: %s\n', figPath);
