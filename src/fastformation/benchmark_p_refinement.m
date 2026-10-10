% BENCHMARK_P_REFINEMENT.M
% Comparison 3: assembly time vs p-refinement (p = 2..7) on a 40 x 20 mesh under a
% random element-wise SIMP density. All numbers are measured on the same density:
%   Normal  - GeoPDEs op_su_ev_tp, (p+1)^2 Gauss points per element, SIMP-scaled
%             Lame coefficients (GEOPDES_PATH; NaN without GeoPDEs)
%   WQ CPU  - per-iteration WQ formation of the sparse K (WQ_FORM), interior layout;
%             one-time setup (WQ_SETUP) reported separately
%   WQ GPU  - same formation on the GPU (FP64), K returned as a host sparse matrix;
%             the on-device kernel time (values left on the device) is also reported

clearvars; clc; close all;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'benchmarks'));
has_gpu = gpuDeviceCount > 0;

p_list = 2:7;
nsub = [40, 20];
rng(42);
xPhys = rand(nsub);
lambda = 0.3 / (1.3 * 0.4); mu = 1 / 2.6;   % E = 1, nu = 0.3

n = numel(p_list);
[p_dofs, t_normal, t_fast, t_setup, t_gpu, t_gpu_kernel, err_gauss] = deal(nan(1, n));
fprintf('%-4s %-6s %-11s %-11s %-11s %-11s %-11s %-9s %-10s\n', 'p', 'DOFs', 'Gauss [s]', ...
    'WQ CPU [s]', 'setup [s]', 'WQ GPU [s]', 'GPU kern', 'Speedup', '|K-Kg|/|Kg|');
for i = 1:n
    p = p_list(i);
    sp = iga_space_box([0 1.0; 0 0.5], nsub, p);
    p_dofs(i) = sp.ndof;
    [t_normal(i), Kg] = geopdes_gauss_baseline([0 1.0; 0 0.5], nsub, p, lambda, mu, xPhys, 3, 1e-3);
    devs = {'cpu'}; if has_gpu, devs{end+1} = 'gpu_fp64'; end %#ok<SAGROW>
    T = wq_time_formation(sp, xPhys, devs, 3);
    t_fast(i) = T.cpu.form; t_setup(i) = T.cpu.setup;
    if has_gpu, t_gpu(i) = T.gpu_fp64.form; t_gpu_kernel(i) = T.gpu_fp64.kernel; end
    if ~isempty(Kg)
        Kw = wq_form(wq_setup(sp, 1, 0.3, 'cpu'), xPhys, 'element', 3, 1e-3, true);
        err_gauss(i) = norm(Kw - Kg, 'fro') / norm(Kg, 'fro');
    end
    fprintf('%-4d %-6d %-11.4f %-11.4f %-11.4f %-11.4f %-11.4f %-9.2f %-10.2e\n', p, p_dofs(i), ...
        t_normal(i), t_fast(i), t_setup(i), t_gpu(i), t_gpu_kernel(i), t_normal(i) / t_fast(i), err_gauss(i));
end

fig = figure('Position', [100, 100, 750, 520], 'Color', 'w', 'InvertHardcopy', 'off', 'Visible', 'off');
semilogy(p_list, t_normal, 'r-o', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', [1 0.8 0.8], 'DisplayName', 'Standard Gauss (GeoPDEs)');
hold on;
semilogy(p_list, t_fast, 'b-s', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', [0.8 0.8 1], 'DisplayName', 'WQ + Sum-Fact (CPU)');
if has_gpu
    semilogy(p_list, t_gpu, 'm-^', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', [1 0.8 1], 'DisplayName', 'WQ + Sum-Fact (GPU, FP64)');
end
grid on;
set(gca, 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'Box', 'on', 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
xlabel('Polynomial Degree p', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Stiffness Formation Time [s]', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
title(sprintf('Formation Time vs. p-Refinement (%d elements, DOFs: %d - %d)', prod(nsub), p_dofs(1), p_dofs(end)), ...
    'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'FontSize', 10, 'TextColor', 'k', 'Box', 'on');
exportgraphics(fig, 'fig_timing_p_refinement.png', 'Resolution', 300);
close(fig);
save('benchmark_p_results.mat', 'p_list', 'p_dofs', 't_normal', 't_fast', 't_setup', 't_gpu', 't_gpu_kernel', 'err_gauss', 'nsub');
