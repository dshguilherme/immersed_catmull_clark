% REPLOT_SCALING_FIGURES
% Clean white backgrounds for axes and legends
clear; clc; close all;
load('benchmark_scaling_results.mat');

%% Figure 1: Scaling Comparison
fig1 = figure('Color', 'w', 'Position', [100, 100, 750, 520], 'Visible', 'off');

loglog(dofs_all, t_std_gauss, 'r--o', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', 'r', ...
    'DisplayName', 'Standard Gauss (GeoPDEs)');
hold on;
loglog(dofs_all, t_cpu_wq, 'b-s', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', 'b', ...
    'DisplayName', 'Fast CPU (WQ + Sum-Fact)');
loglog(dofs_all, t_gpu_fp64, 'm-^', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', 'm', ...
    'DisplayName', 'GPU Matrix-Free (FP64)');
loglog(dofs_all, t_gpu_fp32, 'g-d', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0.7 0], ...
    'DisplayName', 'GPU Matrix-Free (FP32)');
loglog(dofs_all, t_gpu_fp16, 'c-v', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0.8 0.8], ...
    'DisplayName', 'GPU Matrix-Free (FP16 Tensor Cores)');

grid on;
set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on', 'Color', 'w', ...
    'XColor', 'k', 'YColor', 'k', 'GridColor', [0.85 0.85 0.85]);
xlabel('Degrees of Freedom (DOFs)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Operator Evaluation / Assembly Time [s]', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
title('3D Isogeometric Scaling: Standard vs. Fast WQ vs. GPU Precisions', ...
    'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
lgd = legend('Location', 'northwest', 'FontSize', 10.5, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', [0.7 0.7 0.7]);
xlim([400, 150000]);

fig_name1 = 'fig_timing_h_refinement.png';
exportgraphics(fig1, fig_name1, 'Resolution', 300);
fprintf('Re-exported %s with pure white legend\n', fig_name1);

%% Figure 2: Precision Speedup
fig2 = figure('Color', 'w', 'Position', [150, 150, 720, 480], 'Visible', 'off');

speedup_fp32_vs_fp64 = t_gpu_fp64 ./ t_gpu_fp32;
speedup_fp16_vs_fp64 = t_gpu_fp64 ./ t_gpu_fp16;

semilogx(dofs_all, speedup_fp32_vs_fp64, 'g-d', 'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', [0 0.7 0], ...
    'DisplayName', 'FP32 Speedup vs. FP64');
hold on;
semilogx(dofs_all, speedup_fp16_vs_fp64, 'c-v', 'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', [0 0.8 0.8], ...
    'DisplayName', 'FP16 (Tensor Cores) Speedup vs. FP64');
yline(1.0, 'k--', 'LineWidth', 1.5, 'DisplayName', 'FP64 Baseline (1.0x)');

grid on;
set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on', 'Color', 'w', ...
    'XColor', 'k', 'YColor', 'k', 'GridColor', [0.85 0.85 0.85]);
xlabel('Degrees of Freedom (DOFs)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
ylabel('GPU Throughput Speedup relative to FP64', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
title('Reduced-Precision Speedup on NVIDIA RTX 2050 Laptop GPU', ...
    'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
lgd2 = legend('Location', 'northwest', 'FontSize', 10.5, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', [0.7 0.7 0.7]);
xlim([400, 150000]);

fig_name2 = 'fig_precision_speedup.png';
exportgraphics(fig2, fig_name2, 'Resolution', 300);
fprintf('Re-exported %s with pure white legend\n', fig_name2);
