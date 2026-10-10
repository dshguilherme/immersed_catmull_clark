% BENCHMARK_H_REFINEMENT.M
% Comparison 2: Timing of assemblies (Normal vs Fast CPU vs GPU) vs h-refinement.
% Includes random xPhys array and extrapolation for large DOF counts.

clearvars; clc; close all;

fprintf('========================================================================\n');
fprintf('  COMPARISON 2: Assembly Timing vs h-Refinement (p = 3 fixed)\n');
fprintf('  Includes random SIMP density field and large-scale DOF scaling\n');
fprintf('========================================================================\n\n');

p = 3;
nsub_list = {[10, 5], [20, 10], [40, 20], [60, 30], [80, 40], [120, 60], [160, 80], [200, 100], [260, 130]};

h_dofs = zeros(1, numel(nsub_list));
h_nel  = zeros(1, numel(nsub_list));
t_normal = zeros(1, numel(nsub_list));
t_fast   = zeros(1, numel(nsub_list));
t_gpu    = zeros(1, numel(nsub_list));

% Threshold for direct normal assembly measurement (up to ~16k DOFs)
max_normal_mesh_idx = 6; 

fprintf('%-12s %-10s %-12s %-14s %-14s %-14s\n', ...
    'Mesh (nsub)', 'Elements', 'DOFs', 'Normal (s)', 'Fast CPU (s)', 'Fast GPU (s)');
fprintf('%s\n', repmat('-', 1, 80));

rng(42); % Reproducible random seed
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'benchmarks'));
lambda = 0.3 / (1.3 * 0.4); mu = 1 / 2.6;   % E = 1, nu = 0.3

for i = 1:numel(nsub_list)
    nsub = nsub_list{i};
    
    sp = iga_space_box([0 1.0; 0 0.5], nsub, p);

    h_dofs(i) = sp.ndof;
    h_nel(i)  = sp.nel;

    % Random physical density array
    xPhys = rand(nsub(1), nsub(2));

    % 1. Normal GeoPDEs Tensor-Product Assembly (external reference, measured for
    %    i <= max_normal_mesh_idx; NaN when GeoPDEs is not available)
    if i <= max_normal_mesh_idx
        t_normal(i) = geopdes_gauss_baseline([0 1.0; 0 0.5], nsub, p, lambda, mu);
    else
        t_normal(i) = NaN; % Will be extrapolated
    end

    % 2. Fast CPU Assembly
    tic;
    K_f = fast_stiffness_assembly(sp, 1.0, 0.3, xPhys, 'element', 3, 1e-3, true);
    t_fast(i) = toc;

    % 3. Fast GPU Assembly (vectorized device kernel)
    tic;
    [Ke, type_id] = iga_elasticity_element_matrices(sp, lambda, mu);
    [rows, cols] = iga_element_rows_cols(sp.connectivity);
    vals0 = reshape(Ke(:, :, type_id), [], 1);
    n_per_el = sp.nsh^2;
    scale = repelem(1e-3 + (1 - 1e-3)*(xPhys(:).^3), n_per_el);
    vals0_g = gpuArray(vals0); scale_g = gpuArray(scale);
    vals_g = vals0_g .* scale_g;
    vals_cpu = gather(vals_g);
    K_g = sparse(rows, cols, vals_cpu, sp.ndof, sp.ndof);
    t_gpu(i) = toc;
    
    if ~isnan(t_normal(i))
        fprintf('[%3d x %-3d]   %-10d %-12d %-14.4f %-14.4f %-14.4f\n', ...
            nsub(1), nsub(2), h_nel(i), h_dofs(i), t_normal(i), t_fast(i), t_gpu(i));
    else
        fprintf('[%3d x %-3d]   %-10d %-12d [extrapolated] %-14.4f %-14.4f\n', ...
            nsub(1), nsub(2), h_nel(i), h_dofs(i), t_fast(i), t_gpu(i));
    end
end

% Fit power-law curve to measured normal assembly points: t = a * DOFs^b
measured_idx = find(~isnan(t_normal(1:max_normal_mesh_idx)));
if numel(measured_idx) < 2
    warning('benchmark_h_refinement: GeoPDEs baseline not available; normal-assembly curve skipped.');
    measured_idx = 1:max_normal_mesh_idx; t_normal(measured_idx) = NaN;
end
poly_fit = polyfit(log10(h_dofs(measured_idx)), log10(t_normal(measured_idx)), 1);
b_power = poly_fit(1);
a_power = 10^poly_fit(2);

t_normal_extrap = a_power * (h_dofs .^ b_power);

fprintf('\nPower-law fit for Normal Assembly: t = %.2e * N^{%.2f}\n', a_power, b_power);
for i = max_normal_mesh_idx+1 : numel(nsub_list)
    fprintf('  DOFs = %6d : Extrapolated Normal Time = %.2f s (vs Fast CPU = %.2f s, GPU = %.2f s)\n', ...
        h_dofs(i), t_normal_extrap(i), t_fast(i), t_gpu(i));
end

fprintf('%s\n\n', repmat('=', 1, 80));

%% Generate Publication-Quality White-Background Figure
fig = figure('Position', [100, 100, 750, 520], 'Color', 'w', 'InvertHardcopy', 'off');

% Plot measured normal assembly points
loglog(h_dofs(measured_idx), t_normal(measured_idx), 'ro', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.8 0.8], 'DisplayName', 'Normal GeoPDEs (measured)');
hold on;

% Plot extrapolated normal assembly curve
loglog(h_dofs, t_normal_extrap, 'r--', 'LineWidth', 2, ...
    'DisplayName', sprintf('Normal GeoPDEs (extrapolated \\sim N^{%.2f})', b_power));

% Plot Fast CPU assembly
loglog(h_dofs, t_fast, 'b-s', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.8 0.8 1], 'DisplayName', 'Fast CPU (WQ + SumFact)');

% Plot Fast GPU assembly
loglog(h_dofs, t_gpu, 'm-^', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.8 1], 'DisplayName', 'Fast GPU (NVIDIA RTX 2050)');

grid on;
set(gca, 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'Box', 'on', ...
    'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
xlabel('Degrees of Freedom (DOFs)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Assembly Wall-Clock Time [s]', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
title(sprintf('Assembly Time vs. h-Refinement (Degree p = %d, Random SIMP \\rho)', p), ...
    'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'FontSize', 10, 'TextColor', 'k', 'Box', 'on');

fig_name = 'fig_timing_h_refinement.png';
if exist(fig_name, 'file')
    try delete(fig_name); catch; end
end
try
    exportgraphics(fig, fig_name, 'Resolution', 300);
catch
    saveas(fig, fig_name);
end
save('benchmark_h_results.mat', 'h_dofs', 'h_nel', 't_normal', 't_normal_extrap', 't_fast', 't_gpu');
fprintf('Figure saved as %s\n', fig_name);
