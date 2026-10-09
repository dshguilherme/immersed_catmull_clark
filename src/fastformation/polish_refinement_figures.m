% POLISH_REFINEMENT_FIGURES.M
% Generates publication-grade refinement benchmark figures:
% 1. Figure 1 (3D Scaling / H-refinement): Lower axis extended down to 1e-4 to show all GPU precision orders of magnitude without clipping.
% 2. Figure 2 (P-refinement): Removed unnecessary cut speedup text marker.
% 3. Figure 3 (K-refinement): Removed 83x speedup text marker; retained clean point badges and extended axes.
% 4. Pure white legends across all figures with crisp black typography and subtle borders.

clearvars; clc; close all;

%% ========================================================================
%% 1. Figure 1: 3D Scaling Comparison (H-Refinement / Problem Discretization)
%% ========================================================================
load('benchmark_scaling_results.mat');

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

xlim([400, 160000]);
% Lower bound set to 1e-4 so all GPU FP16 and FP32 points are fully visible with generous margins
ylim([1e-4, 500]);

lgd1 = legend('Location', 'northwest', 'FontSize', 10.5);
set(lgd1, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', [0.7 0.7 0.7], 'Box', 'on', 'LineWidth', 1.0);

exportgraphics(fig1, 'fig_timing_h_refinement.png', 'Resolution', 300);
fprintf('Saved polished fig_timing_h_refinement.png (Figure 1, full dynamic range)\n');

%% ========================================================================
%% 2. Figure 2: p-Refinement Timing
%% ========================================================================
load('benchmark_p_results.mat');

fig2 = figure('Color', 'w', 'Position', [100, 100, 750, 520], 'Visible', 'off');

semilogy(p_list, t_normal, 'r-o', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.85 0.85], 'DisplayName', 'Normal GeoPDEs (Full Gauss)');
hold on;
semilogy(p_list, t_fast, 'b-s', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.85 0.85 1], 'DisplayName', 'Fast CPU (WQ + Sum-Fact)');
semilogy(p_list, t_gpu, 'm-^', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.85 1], 'DisplayName', 'Fast GPU (NVIDIA RTX 2050)');

grid on;
set(gca, 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'Box', 'on', ...
    'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
xlabel('Polynomial Degree p', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Operator Assembly Time [s]', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
title(sprintf('Assembly Time vs. p-Refinement (800 Elements, DOFs: %d - %d)', ...
    p_dofs(1), p_dofs(end)), 'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');

xlim([1.7, 7.4]);
ylim([3e-4, 50]);
xticks(2:7);

lgd2 = legend('Location', 'northwest', 'FontSize', 10.5);
set(lgd2, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', [0.7 0.7 0.7], 'Box', 'on', 'LineWidth', 1.0);

exportgraphics(fig2, 'fig_timing_p_refinement.png', 'Resolution', 300);
fprintf('Saved polished fig_timing_p_refinement.png (Figure 2, speedup marker removed)\n');

%% ========================================================================
%% 3. Figure 3: k-Refinement Timing
%% ========================================================================
load('benchmark_k_results.mat');

fig3 = figure('Color', 'w', 'Position', [100, 100, 780, 540], 'Visible', 'off');

loglog(k_dofs, t_normal, 'r-o', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.85 0.85], 'DisplayName', 'Normal GeoPDEs (Full Gauss)');
hold on;
loglog(k_dofs, t_fast, 'b-s', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.85 0.85 1], 'DisplayName', 'Fast CPU (WQ + Sum-Fact)');
loglog(k_dofs, t_gpu, 'm-^', 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.85 1], 'DisplayName', 'Fast GPU (NVIDIA RTX 2050)');

% Extended axes: generous room for all point badges
xlim([90, 60000]);
ylim([2e-4, 600]);

% Polished point badges
n_steps = numel(p_list);
for i = 1:n_steps
    txt = sprintf('p = %d (%dx%d)', p_list(i), mesh_list{i}(1), mesh_list{i}(2));
    
    if i <= 2
        y_pos = t_normal(i) * 2.2;
        x_pos = k_dofs(i);
        ha = 'center';
    elseif i <= 4
        y_pos = t_normal(i) * 2.0;
        x_pos = k_dofs(i) * 0.95;
        ha = 'center';
    else
        y_pos = t_normal(i) * 1.0;
        x_pos = k_dofs(i) * 1.25;
        ha = 'left';
    end
    
    text(x_pos, y_pos, txt, 'FontSize', 8.5, 'FontWeight', 'bold', ...
        'Color', [0.65 0 0], 'HorizontalAlignment', ha, ...
        'BackgroundColor', [1 1 1 0.92], 'Margin', 2.5, ...
        'EdgeColor', [0.82 0.82 0.82], 'LineWidth', 0.8);
end

grid on;
set(gca, 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'Box', 'on', ...
    'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
xlabel('Degrees of Freedom (DOFs)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Operator Assembly Time [s]', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
title('Assembly Time vs. k-Refinement (Simultaneous p and h Advancement, C^{p-1})', ...
    'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');

lgd3 = legend('Location', 'northwest', 'FontSize', 10.5);
set(lgd3, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', [0.7 0.7 0.7], 'Box', 'on', 'LineWidth', 1.0);

exportgraphics(fig3, 'fig_timing_k_refinement.png', 'Resolution', 300);
fprintf('Saved polished fig_timing_k_refinement.png (Figure 3, 83x speedup marker removed)\n');

%% ========================================================================
%% 4. Figure 4: Precision Speedup
%% ========================================================================
fig4 = figure('Color', 'w', 'Position', [150, 150, 720, 480], 'Visible', 'off');

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

xlim([400, 160000]);
ylim([0.8, 3.8]);

lgd4 = legend('Location', 'northwest', 'FontSize', 10.5);
set(lgd4, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', [0.7 0.7 0.7], 'Box', 'on', 'LineWidth', 1.0);

exportgraphics(fig4, 'fig_precision_speedup.png', 'Resolution', 300);
fprintf('Saved polished fig_precision_speedup.png\n');
fprintf('All refinement figures polished and re-rendered successfully!\n');
