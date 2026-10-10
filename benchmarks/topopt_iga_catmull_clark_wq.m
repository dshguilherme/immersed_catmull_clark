function [xPhys, compliance_history, change_history, time_history] = topopt_iga_catmull_clark_wq(nelx, nely, volfrac, penal, rmin, max_iter, device, precision)
% TOPOPT_IGA_CATMULL_CLARK_WQ
% High-Performance 2D Cantilever Topology Optimization using
% Catmull-Clark limit basis with FastFormation Batched Weighted Quadrature (WQ).
%
% Integrates:
%   1. Catmull-Clark cubic limit basis projection & adjoint sensitivity backpropagation.
%   2. FastFormation WQ sum-factorization kernel (GPU / CPU, FP64 / FP32).
%   3. Preconditioned Conjugate Gradient (PCG) or Direct sparse solve.

if nargin < 1 || isempty(nelx), nelx = 80; end
if nargin < 2 || isempty(nely), nely = 40; end
if nargin < 3 || isempty(volfrac), volfrac = 0.5; end
if nargin < 4 || isempty(penal), penal = 3.0; end
if nargin < 5 || isempty(rmin), rmin = 2.0; end
if nargin < 6 || isempty(max_iter), max_iter = 50; end
if nargin < 7 || isempty(device), device = 'gpu'; end
if nargin < 8 || isempty(precision), precision = 'double'; end

% Paths
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src', 'iga'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src', 'catmull_clark'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src', 'fastformation'));

fprintf('========================================================================\n');
fprintf('  CATMULL-CLARK IGA WITH BATCHED WEIGHTED QUADRATURE (WQ)\n');
fprintf('  Mesh: %d x %d | Device: %s (%s) | VolFrac: %.2f | Penal: %.1f\n', ...
    nelx, nely, upper(device), precision, volfrac, penal);
fprintf('========================================================================\n\n');

%% 1. Problem Geometry and Discrete Spaces
L = 1.0; h = 0.5;
degree = 3;
space = iga_space_box([0 L; 0 h], [nelx, nely], degree);
free_dofs = setdiff(1:space.ndof, space.boundary(1).dofs);

% External Force Vector (normalized distributed load on center-right edge)
F = iga_load_vector(space, @(x, y) iga_cantilever_tip_load(x, y, L, h, 'center'));
Fy_tot = abs(sum(F(space.ndof_sc + 1 : end)));
if Fy_tot > 0
    F = F / Fy_tot;
end

%% 2. Setup Batched Weighted Quadrature (WQ) Operator
% One-time geometry-dependent setup; independent of design density
t_setup = tic;
YOUNG = 1.0; POISSON = 0.3;
S_wq = wq_setup(space, YOUNG, POISSON, device, precision);
t_setup_time = toc(t_setup);
fprintf('WQ Batched Operator initialized in %.2f s (Device: %s)\n\n', ...
    t_setup_time, device);

%% 3. Catmull-Clark Limit Control Point Lattice
% On regular quadrilateral mesh, Catmull-Clark control point net has
% dimensions ncp_x x ncp_y with C^2 continuity
ncp_x = space.ndof_dir(1);
ncp_y = space.ndof_dir(2);
n_vars = ncp_x * ncp_y;

% Initial uniform distribution for Catmull-Clark control points
xPhys = volfrac * ones(ncp_x, ncp_y);

% Evaluation of 1D Catmull-Clark bases at element centers for volume evaluation
xc = 0.5 * (space.breaks{1}(1:end-1) + space.breaks{1}(2:end));
yc = 0.5 * (space.breaks{2}(1:end-1) + space.breaks{2}(2:end));
P1 = iga_bspline_basis(space.knots{1}, degree, xc);
P2 = iga_bspline_basis(space.knots{2}, degree, yc);

V_target = volfrac * (nelx * nely);
V_eval = @(x) sum(sum(P1 * x * (P2')));

Emin = 1e-3;

compliance_history = zeros(max_iter, 1);
change_history = zeros(max_iter, 1);
time_history = zeros(max_iter, 1);

fprintf('%-6s %-16s %-14s %-14s %-12s\n', ...
    'Iter', 'Compliance', 'VolFrac', 'Change', 'Time (s)');
fprintf('%s\n', repmat('-', 1, 68));

change = 1.0;
iter = 0;
move = 0.2;

%% 4. Optimization Loop
while iter < max_iter && change > 1e-3
    iter = iter + 1;
    t_iter = tic;
    x_old = xPhys;
    
    % --- Step 1: Form Stiffness via Batched WQ Kernel in Spline Mode ---
    % WQ_FORM computes 1D sum-factorization directly from the Catmull-Clark control net
    K = wq_form(S_wq, xPhys, 'spline', penal, Emin, true, 'cpu_sparse');
    
    % --- Step 2: State Solution ---
    U = zeros(space.ndof, 1);
    U(free_dofs) = K(free_dofs, free_dofs) \ F(free_dofs);
    c = F' * U;
    compliance_history(iter) = c;
    
    % --- Step 3: Sensitivity Formulation via Fast Sensitivities ---
    % Evaluates exact sensitivities at Catmull-Clark control points
    dC_cp = fast_sensitivities(U, space, xPhys, 'spline', penal, Emin, YOUNG, POISSON);
    sens = -dC_cp;
    
    % --- Step 4: Optimality Criteria Bisection Search ---
    l1 = 0;
    l2 = max(sens(:)) * 2;
    while V_eval(max(1e-3, max(x_old - move, min(1.0, min(x_old + move, x_old .* sqrt(max(0, sens / l2))))))) > V_target
        l2 = l2 * 2;
    end
    
    while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4
        lmid = 0.5 * (l1 + l2);
        Be = max(0, sens / lmid);
        x_new = max(1e-3, max(x_old - move, min(1.0, min(x_old + move, x_old .* sqrt(Be)))));
        if V_eval(x_new) > V_target
            l1 = lmid;
        else
            l2 = lmid;
        end
    end
    
    change = max(abs(x_new(:) - x_old(:)));
    xPhys = x_new;
    curr_vol = V_eval(xPhys) / (nelx * nely);
    
    change_history(iter) = change;
    time_history(iter) = toc(t_iter);
    
    fprintf('%-6d %-16.4f %-14.4f %-14.4e %-12.3f\n', ...
        iter, c, curr_vol, change, time_history(iter));
end

fprintf('%s\n', repmat('=', 1, 68));
fprintf('WQ Catmull-Clark Optimization completed in %d iterations (Total time: %.2f s)\n\n', ...
    iter, sum(time_history(1:iter)));

compliance_history = compliance_history(1:iter);
change_history = change_history(1:iter);
time_history = time_history(1:iter);

%% 5. Generate Publication-Quality Figure
figDir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end

fig = figure('Position', [100, 100, 1000, 420], 'Color', 'w', 'InvertHardcopy', 'off');

subplot(1, 2, 1);
plot_density = (P1 * xPhys * (P2'))';
imagesc([0 L], [0 h], plot_density);
colormap(flipud(gray));
axis equal tight;
set(gca, 'YDir', 'normal', 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'Box', 'on', 'LineWidth', 1.2);
try clim([0 1]); catch, caxis([0 1]); end
cb = colorbar;
set(cb, 'Color', 'k', 'FontSize', 10);
ylabel(cb, 'Physical Density \rho', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
title(sprintf('Optimal Topology (Catmull-Clark WQ, %dx%d)', nelx, nely), ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
xlabel('x / L', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('y / h', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

subplot(1, 2, 2);
yyaxis left;
plot(1:iter, compliance_history, 'b-o', 'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'b');
ylabel('Compliance J = F^T U', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'b');
set(gca, 'YColor', 'b');

yyaxis right;
semilogy(1:iter, change_history, 'r--s', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'r');
ylabel('Max Change \Delta \rho_{max}', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r');
set(gca, 'YColor', 'r');

grid on;
set(gca, 'FontSize', 11, 'XColor', 'k', 'Color', 'w', 'Box', 'on', 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
title('Convergence History (WQ & Catmull-Clark)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Iteration', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

figPath = fullfile(figDir, 'fig_topopt_catmull_clark_wq_2d.png');
exportgraphics(fig, figPath, 'Resolution', 150);
close(fig);
fprintf('Saved figure: %s\n', figPath);

end
