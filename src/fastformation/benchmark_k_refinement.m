% BENCHMARK_K_REFINEMENT.M
% Comparison 4: assembly time vs k-refinement: p = 2..7 with C^{p-1} continuity and
% meshes 10x5 .. 100x50, random element-wise SIMP density. Same measured quantities
% as BENCHMARK_P_REFINEMENT (GeoPDEs Gauss, WQ CPU formation, WQ GPU formation).

clearvars; clc; close all;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'benchmarks'));
has_gpu = gpuDeviceCount > 0;

p_list = 2:7;
mesh_list = {[10, 5], [20, 10], [40, 20], [60, 30], [80, 40], [100, 50]};
rng(42);
lambda = 0.3 / (1.3 * 0.4); mu = 1 / 2.6;

n = numel(p_list);
[k_dofs, k_nel, t_normal, t_fast, t_setup, t_gpu, t_gpu_kernel] = deal(nan(1, n));
fprintf('%-4s %-9s %-6s %-11s %-11s %-11s %-11s %-11s %-9s\n', 'p', 'Mesh', 'DOFs', 'Gauss [s]', ...
    'WQ CPU [s]', 'setup [s]', 'WQ GPU [s]', 'GPU kern', 'Speedup');
for i = 1:n
    p = p_list(i); nsub = mesh_list{i};
    sp = iga_space_box([0 1.0; 0 0.5], nsub, p);
    k_dofs(i) = sp.ndof; k_nel(i) = sp.nel;
    xPhys = rand(nsub);
    t_normal(i) = geopdes_gauss_baseline([0 1.0; 0 0.5], nsub, p, lambda, mu, xPhys, 3, 1e-3);
    devs = {'cpu'}; if has_gpu, devs{end+1} = 'gpu_fp64'; end %#ok<SAGROW>
    T = wq_time_formation(sp, xPhys, devs, 3);
    t_fast(i) = T.cpu.form; t_setup(i) = T.cpu.setup;
    if has_gpu, t_gpu(i) = T.gpu_fp64.form; t_gpu_kernel(i) = T.gpu_fp64.kernel; end
    fprintf('%-4d %3dx%-5d %-6d %-11.4f %-11.4f %-11.4f %-11.4f %-11.4f %-9.2f\n', p, nsub(1), nsub(2), ...
        k_dofs(i), t_normal(i), t_fast(i), t_setup(i), t_gpu(i), t_gpu_kernel(i), t_normal(i) / t_fast(i));
end

fig = figure('Position', [100, 100, 780, 540], 'Color', 'w', 'InvertHardcopy', 'off', 'Visible', 'off');
loglog(k_dofs, t_normal, 'r-o', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', [1 0.8 0.8], 'DisplayName', 'Standard Gauss (GeoPDEs)');
hold on;
loglog(k_dofs, t_fast, 'b-s', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', [0.8 0.8 1], 'DisplayName', 'WQ + Sum-Fact (CPU)');
if has_gpu
    loglog(k_dofs, t_gpu, 'm-^', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', [1 0.8 1], 'DisplayName', 'WQ + Sum-Fact (GPU, FP64)');
end
for i = 1:n
    text(k_dofs(i) * 1.08, t_normal(i), sprintf(' p=%d\n [%dx%d]', p_list(i), mesh_list{i}(1), mesh_list{i}(2)), ...
        'FontSize', 8.5, 'Color', [0.6 0 0], 'FontWeight', 'bold');
end
grid on;
set(gca, 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'Box', 'on', 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
xlabel('Degrees of Freedom (DOFs)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Stiffness Formation Time [s]', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
title('Formation Time vs. k-Refinement (Simultaneous p and h Elevation, C^{p-1})', 'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'FontSize', 10, 'TextColor', 'k', 'Box', 'on');
xlim([k_dofs(1) * 0.7, k_dofs(end) * 2.0]);
exportgraphics(fig, 'fig_timing_k_refinement.png', 'Resolution', 300);
close(fig);
save('benchmark_k_results.mat', 'p_list', 'mesh_list', 'k_dofs', 'k_nel', 't_normal', 't_fast', 't_setup', 't_gpu', 't_gpu_kernel');
