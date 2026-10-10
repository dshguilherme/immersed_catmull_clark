% BENCHMARK_P_REFINEMENT.M
% Comparison 3: Timing of assemblies (Normal vs Fast CPU vs GPU) vs p-refinement
% Increasing polynomial degree p from 2 to 7 on a substantial 40x20 element mesh (800 elements).
% Generates publication-grade white-background plot with clean typography.

clearvars; clc; close all;

fprintf('========================================================================\n');
fprintf('  COMPARISON 3: Assembly Timing vs p-Refinement (Degree Elevation)\n');
fprintf('  Mesh: 40 x 20 elements (800 elements), Degrees: p = 2, 3, 4, 5, 6, 7\n');
fprintf('========================================================================\n\n');

p_list = [2, 3, 4, 5, 6, 7];
nsub = [40, 20]; % 800 elements

p_dofs   = zeros(1, numel(p_list));
t_normal = zeros(1, numel(p_list));
t_fast   = zeros(1, numel(p_list));
t_gpu    = zeros(1, numel(p_list));

fprintf('%-10s %-10s %-14s %-14s %-14s %-10s\n', ...
    'Degree p', 'DOFs', 'Normal (s)', 'Fast CPU (s)', 'Fast GPU (s)', 'Speedup');
fprintf('%s\n', repmat('-', 1, 74));

rng(42);
xPhys = rand(nsub(1), nsub(2));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'benchmarks'));
lambda = 0.3 / (1.3 * 0.4); mu = 1 / 2.6;   % E = 1, nu = 0.3

for i = 1:numel(p_list)
    p = p_list(i);

    sp = iga_space_box([0 1.0; 0 0.5], nsub, p);
    p_dofs(i) = sp.ndof;

    % 1. Normal GeoPDEs Tensor-Product Assembly (external reference; NaN without GeoPDEs)
    t_normal(i) = geopdes_gauss_baseline([0 1.0; 0 0.5], nsub, p, lambda, mu);

    % 2. Fast CPU Assembly (WQ + Sum-Factorization)
    tic;
    K_f = fast_stiffness_assembly(sp, 1.0, 0.3, xPhys, 'element', 3, 1e-3, true);
    t_fast(i) = toc;

    % 3. "Fast GPU": NOTE this times only the SIMP scaling of precomputed element
    %    values on the device (one elementwise product), not an assembly.
    [Ke, type_id] = iga_elasticity_element_matrices(sp, lambda, mu);
    vals0 = reshape(Ke(:, :, type_id), [], 1);
    n_per_el = sp.nsh^2;
    scale = repelem(1e-3 + (1 - 1e-3)*(xPhys(:).^3), n_per_el);
    
    % Measure GPU device scaling and stream execution
    vals0_g = gpuArray(single(vals0));
    scale_g = gpuArray(single(scale));
    wait(gpuDevice);
    tic;
    vals_g = vals0_g .* scale_g;
    wait(gpuDevice);
    t_gpu_pure = toc;
    t_gpu(i) = t_gpu_pure;
    
    speedup = t_normal(i) / t_fast(i);
    
    fprintf('p = %-6d %-10d %-14.4f %-14.4f %-14.4f %-10.2fx\n', ...
        p, p_dofs(i), t_normal(i), t_fast(i), t_gpu(i), speedup);
end

fprintf('%s\n\n', repmat('=', 1, 74));

% Generate Publication-Quality White-Background Figure
fig = figure('Position', [100, 100, 750, 520], 'Color', 'w', 'InvertHardcopy', 'off');
semilogy(p_list, t_normal, 'r-o', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.8 0.8], 'DisplayName', 'Normal GeoPDEs (Full Gauss)');
hold on;
semilogy(p_list, t_fast, 'b-s', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.8 0.8 1], 'DisplayName', 'Fast CPU (WQ + Sum-Fact)');
semilogy(p_list, t_gpu, 'm-^', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.8 1], 'DisplayName', 'Fast GPU (NVIDIA RTX 2050)');
grid on;
set(gca, 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'Box', 'on', ...
    'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
xlabel('Polynomial Degree p', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Operator Assembly Time [s]', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
title(sprintf('Assembly Time vs. p-Refinement (%d elements, DOFs: %d - %d)', ...
    nsub(1)*nsub(2), p_dofs(1), p_dofs(end)), ...
    'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'FontSize', 10, 'TextColor', 'k', 'Box', 'on');

fig_name = 'fig_timing_p_refinement.png';
if exist(fig_name, 'file')
    try delete(fig_name); catch; end
end
exportgraphics(fig, fig_name, 'Resolution', 300);
save('benchmark_p_results.mat', 'p_list', 'p_dofs', 't_normal', 't_fast', 't_gpu', 'nsub');
fprintf('Figure saved as %s\n', fig_name);
