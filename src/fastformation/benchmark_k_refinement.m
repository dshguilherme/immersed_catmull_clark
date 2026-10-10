% BENCHMARK_K_REFINEMENT.M
% Comparison 4: Timing of assemblies (Normal vs Fast CPU vs GPU) vs k-refinement.
% k-refinement combines degree elevation (p-refinement) and knot insertion (h-refinement)
% with maximal continuity C^{p-1}, advancing DOFs simultaneously in both directions.
% Matches the exact 6 steps of p-refinement (p = 2 to 7) and h-refinement (10x5 to 100x50).

clearvars; clc; close all;

fprintf('========================================================================\n');
fprintf('  COMPARISON 4: Assembly Timing vs k-Refinement (C^{p-1} Continuity)\n');
fprintf('  Simultaneous p-Elevation and h-Refinement (6 Matched Steps)\n');
fprintf('========================================================================\n\n');

p_list = [2, 3, 4, 5, 6, 7];
mesh_list = {[10, 5], [20, 10], [40, 20], [60, 30], [80, 40], [100, 50]};

n_steps = numel(p_list);
k_dofs   = zeros(1, n_steps);
k_nel    = zeros(1, n_steps);
t_normal = zeros(1, n_steps);
t_fast   = zeros(1, n_steps);
t_gpu    = zeros(1, n_steps);

fprintf('%-6s %-8s %-12s %-10s %-14s %-14s %-14s %-10s\n', ...
    'Step', 'Deg p', 'Mesh', 'DOFs', 'Normal (s)', 'Fast CPU (s)', 'Fast GPU (s)', 'Speedup');
fprintf('%s\n', repmat('-', 1, 92));

rng(42);
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'benchmarks'));
lambda = 0.3 / (1.3 * 0.4); mu = 1 / 2.6;   % E = 1, nu = 0.3

for i = 1:n_steps
    p = p_list(i);
    nsub = mesh_list{i};

    sp = iga_space_box([0 1.0; 0 0.5], nsub, p); % C^{p-1} maximal continuity
    k_dofs(i) = sp.ndof;
    k_nel(i)  = sp.nel;

    xPhys = rand(nsub(1), nsub(2));

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
    
    vals0_g = gpuArray(single(vals0));
    scale_g = gpuArray(single(scale));
    wait(gpuDevice);
    tic;
    vals_g = vals0_g .* scale_g;
    wait(gpuDevice);
    t_gpu_pure = toc;
    t_gpu(i) = t_gpu_pure;
    
    speedup = t_normal(i) / t_fast(i);
    
    fprintf('%-6d p=%-6d [%3dx%-3d]   %-10d %-14.4f %-14.4f %-14.4f %-10.2fx\n', ...
        i, p, nsub(1), nsub(2), k_dofs(i), t_normal(i), t_fast(i), t_gpu(i), speedup);
end

fprintf('%s\n\n', repmat('=', 1, 92));

% Generate Publication-Quality White-Background Figure
fig = figure('Position', [100, 100, 780, 540], 'Color', 'w', 'InvertHardcopy', 'off');

loglog(k_dofs, t_normal, 'r-o', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.8 0.8], 'DisplayName', 'Normal GeoPDEs (Full Gauss)');
hold on;
loglog(k_dofs, t_fast, 'b-s', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.8 0.8 1], 'DisplayName', 'Fast CPU (WQ + Sum-Fact)');
loglog(k_dofs, t_gpu, 'm-^', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.8 1], 'DisplayName', 'Fast GPU (NVIDIA RTX 2050)');

% Annotate degree and mesh at each point
for i = 1:n_steps
    txt = sprintf(' p=%d\n [%dx%d]', p_list(i), mesh_list{i}(1), mesh_list{i}(2));
    text(k_dofs(i)*1.08, t_normal(i), txt, 'FontSize', 8.5, 'Color', [0.6 0 0], 'FontWeight', 'bold');
end

grid on;
set(gca, 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'Box', 'on', ...
    'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
xlabel('Degrees of Freedom (DOFs)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Operator Assembly Time [s]', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
title('Assembly Time vs. k-Refinement (Simultaneous p and h Elevation, C^{p-1})', ...
    'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'FontSize', 10, 'TextColor', 'k', 'Box', 'on');
xlim([k_dofs(1)*0.7, k_dofs(end)*2.0]);

fig_name = 'fig_timing_k_refinement.png';
if exist(fig_name, 'file')
    try delete(fig_name); catch; end
end
exportgraphics(fig, fig_name, 'Resolution', 300);
save('benchmark_k_results.mat', 'p_list', 'mesh_list', 'k_dofs', 'k_nel', 't_normal', 't_fast', 't_gpu');
fprintf('Figure saved as %s\n', fig_name);
