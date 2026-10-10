function [xPhys, compliance_history, time_history] = topopt_iga_fast(nelx, nely, volfrac, penal, rmin, max_iter, density_type, degree)
% TOPOPT_IGA_FAST High-Performance Isogeometric Topology Optimization
% using Fast Matrix Formation and Exact Direct Strain Sensitivities.
%
% Inputs:
%   nelx         - Number of elements along X (default: 80)
%   nely         - Number of elements along Y (default: 40)
%   volfrac      - Target volume fraction (default: 0.5)
%   penal        - SIMP penalty exponent (default: 3.0)
%   rmin         - Filter radius in element units (default: 2.0)
%   max_iter     - Maximum number of iterations (default: 60)
%   density_type - 'element' (default) or 'spline'
%   degree       - B-spline polynomial degree p (default: 3)

if nargin < 1 || isempty(nelx), nelx = 80; end
if nargin < 2 || isempty(nely), nely = 40; end
if nargin < 3 || isempty(volfrac), volfrac = 0.5; end
if nargin < 4 || isempty(penal), penal = 3.0; end
if nargin < 5 || isempty(rmin), rmin = 2.0; end
if nargin < 6 || isempty(max_iter), max_iter = 60; end
if nargin < 7 || isempty(density_type), density_type = 'element'; end
if nargin < 8 || isempty(degree), degree = 3; end

fprintf('========================================================================\n');
fprintf('  ISOGEOMETRIC TOPOLOGY OPTIMIZATION (HIGH-PERFORMANCE KERNEL)\n');
fprintf('  Mesh: %d x %d | Degree: p = %d (C^{%d}) | VolFrac: %.2f | Type: %s\n', ...
    nelx, nely, degree, degree-1, volfrac, density_type);
fprintf('========================================================================\n\n');

%% 1. Problem Setup and Mesh Construction
L = 1.0; h = 0.5;
E0 = 1; nu = 0.3;
lambda = nu * E0 / ((1 + nu) * (1 - 2 * nu));
mu = E0 / (2 * (1 + nu));
space = iga_space_box([0 L; 0 h], [nelx, nely], degree);

% Boundary DOFs (clamped on left face x = 0)
free_dofs = setdiff(1:space.ndof, space.boundary(1).dofs);

% External Force Vector (normalized distributed load on center-right edge)
F = iga_load_vector(space, @(x, y) iga_cantilever_tip_load(x, y, L, h, 'center'));
Fy_tot = abs(sum(F(space.ndof_sc + 1 : end)));
if Fy_tot > 0
    F = F / Fy_tot;
end

%% 2. Fast Precomputed Element Stiffness Operator
t_pre = tic;
[Ke, type_id] = iga_elasticity_element_matrices(space, lambda, mu);
[rows, cols] = iga_element_rows_cols(space.connectivity);
vals0 = reshape(Ke(:, :, type_id), [], 1);
n_per_el = space.nsh^2;
t_pre_time = toc(t_pre);
fprintf('Operator precomputed in %.2f s (DOFs: %d, Non-zeros: %d)\n\n', ...
    t_pre_time, space.ndof, numel(vals0));

Emin = 1e-3;

%% 3. Setup Parameterization and Projection / Filtering
if strcmpi(density_type, 'element')
    % Option A: Piecewise constant element densities
    n_vars = nelx * nely;
    xPhys = volfrac * ones(nelx, nely);
    V_target = volfrac * n_vars;
    
    % Sensitivity filter matrix H
    [Xe, Ye] = meshgrid(1:nelx, 1:nely);
    Xe = Xe'; Ye = Ye';
    Xv = Xe(:); Yv = Ye(:);
    iH = []; jH = []; sH = [];
    for i = 1:n_vars
        dist = sqrt((Xv - Xv(i)).^2 + (Yv - Yv(i)).^2);
        nbrs = find(dist < rmin);
        weights = rmin - dist(nbrs);
        iH = [iH; repmat(i, numel(nbrs), 1)];
        jH = [jH; nbrs];
        sH = [sH; weights];
    end
    H = sparse(iH, jH, sH, n_vars, n_vars);
    Hs = sum(H, 2);
    
else
    % Option B: Continuous B-spline parameterization
    ncp_x = space.ndof_dir(1);
    ncp_y = space.ndof_dir(2);
    n_vars = ncp_x * ncp_y;
    xPhys = volfrac * ones(ncp_x, ncp_y);

    % Evaluation of 1D B-spline bases at element centers
    xc = 0.5 * (space.breaks{1}(1:end-1) + space.breaks{1}(2:end));
    yc = 0.5 * (space.breaks{2}(1:end-1) + space.breaks{2}(2:end));

    P1 = iga_bspline_basis(space.knots{1}, degree, xc); % [nelx x ncp_x]
    P2 = iga_bspline_basis(space.knots{2}, degree, yc); % [nely x ncp_y]
    
    V_target = volfrac * (nelx * nely);
end

compliance_history = zeros(max_iter, 1);
change_history = zeros(max_iter, 1);
time_history = zeros(max_iter, 1);

fprintf('%-6s %-16s %-14s %-14s %-12s\n', ...
    'Iter', 'Compliance', 'VolFrac', 'Change', 'Time (s)');
fprintf('%s\n', repmat('-', 1, 68));

%% 4. Optimization Loop (Optimality Criteria)
change = 1.0;
iter = 0;
move = 0.2;

while iter < max_iter && change > 1e-3
    iter = iter + 1;
    t_iter = tic;
    
    % --- Step 1: Mapping to Physical Element Densities ---
    if strcmpi(density_type, 'element')
        rho_e = xPhys(:);
    else
        rho_grid = P1 * xPhys * (P2');
        rho_e = rho_grid(:);
    end
    
    % --- Step 2: High-Performance Stiffness Assembly ---
    scale = repelem(Emin + (1 - Emin) * (rho_e.^penal), n_per_el);
    K = sparse(rows, cols, vals0 .* scale, space.ndof, space.ndof);
    
    % --- Step 3: State Solution ---
    U = zeros(space.ndof, 1);
    U(free_dofs) = K(free_dofs, free_dofs) \ F(free_dofs);
    
    % --- Step 4: Compliance & Energy Evaluation ---
    c = F' * U;
    compliance_history(iter) = c;
    
    ce = sum(reshape(U(rows) .* vals0 .* U(cols), n_per_el, space.nel), 1)';
    dC_elem = - penal * (1 - Emin) * (rho_e .^ (penal - 1)) .* ce;
    
    % --- Step 5: Sensitivities Formulation & Filtering ---
    if strcmpi(density_type, 'element')
        x_vec = xPhys(:);
        dC_filt = reshape((H * (x_vec .* dC_elem)) ./ (Hs .* max(1e-3, x_vec)), [nelx, nely]);
        sens = -dC_filt;
        x_old = xPhys;
        V_eval = @(x) sum(x(:));
    else
        % Adjoint backpropagation to control point variables
        dC_cp = P1' * reshape(dC_elem, [nelx, nely]) * P2;
        sens = -dC_cp;
        x_old = xPhys;
        V_eval = @(x) sum(sum(P1 * x * (P2')));
    end
    
    % --- Step 6: Optimality Criteria Bisection Search ---
    l1 = 0;
    l2 = max(sens(:)) * 2;
    % Ensure valid bracket
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
fprintf('Optimization completed in %d iterations (Total time: %.2f s)\n\n', iter, sum(time_history(1:iter)));

%% 5. Publication-Quality White-Background Visualizations
fig = figure('Position', [100, 100, 1000, 420], 'Color', 'w', 'InvertHardcopy', 'off');

subplot(1, 2, 1);
if strcmpi(density_type, 'element')
    plot_density = xPhys';
else
    plot_density = (P1 * xPhys * (P2'))';
end

imagesc([0 L], [0 h], plot_density);
colormap(flipud(gray));
axis equal tight;
set(gca, 'YDir', 'normal', 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'Box', 'on', 'LineWidth', 1.2);
try clim([0 1]); catch, caxis([0 1]); end
cb = colorbar;
set(cb, 'Color', 'k', 'FontSize', 10);
ylabel(cb, 'Physical Density \rho', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
title(sprintf('Optimal Topology (%s, p = %d, %dx%d)', density_type, degree, nelx, nely), ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
xlabel('x / L', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('y / h', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

subplot(1, 2, 2);
yyaxis left;
plot(1:iter, compliance_history(1:iter), 'b-o', 'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'b');
ylabel('Compliance J = F^T U', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'b');
set(gca, 'YColor', 'b');

yyaxis right;
semilogy(1:iter, change_history(1:iter), 'r--s', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'r');
ylabel('Max Change \Delta \rho_{max}', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r');
set(gca, 'YColor', 'r');

grid on;
set(gca, 'FontSize', 11, 'XColor', 'k', 'Color', 'w', 'Box', 'on', 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
title('Convergence History (Compliance & Change)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Iteration', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

saveas(fig, sprintf('fig_topopt_%s_p%d.png', density_type, degree));
fprintf('Saved publication topology plot as fig_topopt_%s_p%d.png\n', density_type, degree);

end
