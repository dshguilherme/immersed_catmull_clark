% BENCHMARK_WALLCLOCK_SCALING
% 3D wall-clock time vs DOFs (p = 2, C^1, box 1.2 x 0.6 x 0.3, random element-wise
% SIMP density), 540 .. 101,184 DOFs. Every reported number is a measurement:
%   1. Standard Gauss (GeoPDEs op_su_ev_tp, 27 points/element, SIMP coefficients):
%      measured at every size; a size is skipped (NaN) only if the previous one
%      took longer than gauss_cap seconds (no extrapolation).
%   2. WQ CPU: per-iteration WQ formation of the sparse K (WQ_FORM, interior layout).
%   3. WQ GPU FP64 / FP32: the same formation on the GPU, K returned as a host
%      sparse matrix; the on-device kernel time is stored as well.
%   4. GPU element matrix-free apply v = K(rho) p, FP64 / FP32 (the operator used by
%      the PCG solvers of TOPOPT_IGA_3D: exact p = 2 element matrices x SIMP factor;
%      one matrix-vector product, not a formation of K).
% FP16 is not reported: MATLAB gpuArray has no half-precision arithmetic.

clear; clc; close all;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'benchmarks'));
has_geopdes = geopdes_baseline_available();
has_gpu = gpuDeviceCount > 0;
gauss_cap = 900;

mesh_configs = [8 4 2; 16 8 4; 24 12 6; 32 16 8; 44 22 11; 60 30 15];
L = 1.2; h = 0.6; w = 0.3; p = 2;
lambda = 0.3 / (1.3 * 0.4); mu = 1 / 2.6;
nc = size(mesh_configs, 1);
[dofs_all, nel_all, nnz_all, t_std_gauss, t_cpu_wq, t_cpu_setup, t_gpu_wq64, t_gpu_wq32, ...
    t_gpu_kern64, t_gpu_kern32, t_mf64, t_mf32, err32] = deal(nan(nc, 1));
rng(7);
for i = 1:nc
    nel_dir = mesh_configs(i, :);
    sp = iga_space_box([0 L; 0 h; 0 w], nel_dir, p);
    dofs_all(i) = sp.ndof; nel_all(i) = sp.nel;
    rho = rand(nel_dir);
    fprintf('--- %dx%dx%d: %d elements, %d DOFs ---\n', nel_dir, sp.nel, sp.ndof);

    % 1. Standard Gauss (GeoPDEs)
    if has_geopdes && (i == 1 || ~(t_std_gauss(i-1) > gauss_cap))
        t_std_gauss(i) = geopdes_gauss_baseline([0 L; 0 h; 0 w], nel_dir, p, lambda, mu, rho, 3, 1e-3);
    end
    fprintf('  Standard Gauss (GeoPDEs): %.4f s\n', t_std_gauss(i));

    % 2./3. WQ formation (CPU, GPU FP64, GPU FP32)
    devs = {'cpu'}; if has_gpu, devs = [devs, {'gpu_fp64', 'gpu_fp32'}]; end
    T = wq_time_formation(sp, rho, devs, 3);
    t_cpu_wq(i) = T.cpu.form; t_cpu_setup(i) = T.cpu.setup; nnz_all(i) = T.cpu.nnz;
    fprintf('  WQ CPU formation:   %.4f s (setup %.3f s), nnz(K) = %d\n', t_cpu_wq(i), t_cpu_setup(i), nnz_all(i));
    if has_gpu
        t_gpu_wq64(i) = T.gpu_fp64.form; t_gpu_kern64(i) = T.gpu_fp64.kernel;
        t_gpu_wq32(i) = T.gpu_fp32.form; t_gpu_kern32(i) = T.gpu_fp32.kernel; err32(i) = T.gpu_fp32.relerr;
        fprintf('  WQ GPU FP64: %.4f s (kernel %.4f s) | FP32: %.4f s (kernel %.4f s, |K32-K64|/|K64| = %.1e)\n', ...
            t_gpu_wq64(i), t_gpu_kern64(i), t_gpu_wq32(i), t_gpu_kern32(i), err32(i));

        % 4. GPU element matrix-free apply
        [Ke, tid] = iga_elasticity_element_matrices(sp, lambda, mu);
        nsh = size(Ke, 1); nel = sp.nel;
        [r, ~] = iga_element_rows_cols(sp.connectivity);
        conn = reshape(r(1:nsh:end), nsh, nel);          % element dof lists (column of rows)
        scale = 1e-3 + (1 - 1e-3) * rho(:).^3;
        for prec = {'double', 'single'}
            reset(gpuDevice);
            vals = gpuArray(cast(Ke(:, :, tid), prec{1}));
            sc = gpuArray(cast(reshape(scale, 1, 1, []), prec{1}));
            cg = gpuArray(int32(conn)); pv = gpuArray(cast(rand(sp.ndof, 1), prec{1}));
            y = accumarray(cg(:), reshape(pagemtimes(vals, reshape(pv(cg), nsh, 1, nel)) .* sc, [], 1), [sp.ndof, 1]); %#ok<NASGU>
            wait(gpuDevice); t0 = tic;
            for rep = 1:20
                y = accumarray(cg(:), reshape(pagemtimes(vals, reshape(pv(cg), nsh, 1, nel)) .* sc, [], 1), [sp.ndof, 1]); %#ok<NASGU>
            end
            wait(gpuDevice); tm = toc(t0) / 20;
            if strcmp(prec{1}, 'double'), t_mf64(i) = tm; else, t_mf32(i) = tm; end
            clear vals sc cg pv y
        end
        fprintf('  GPU element matrix-free apply: FP64 %.2f ms | FP32 %.2f ms\n', 1e3 * t_mf64(i), 1e3 * t_mf32(i));
    end
end

save('benchmark_scaling_results.mat', 'dofs_all', 'nel_all', 'nnz_all', 'mesh_configs', 't_std_gauss', ...
    't_cpu_wq', 't_cpu_setup', 't_gpu_wq64', 't_gpu_wq32', 't_gpu_kern64', 't_gpu_kern32', 't_mf64', 't_mf32', 'err32');

fig = figure('Color', 'w', 'Position', [100, 100, 750, 520], 'Visible', 'off');
loglog(dofs_all, t_std_gauss, 'r--o', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Standard Gauss (GeoPDEs)');
hold on;
loglog(dofs_all, t_cpu_wq, 'b-s', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', 'WQ + Sum-Fact formation (CPU)');
if has_gpu
    loglog(dofs_all, t_gpu_wq64, 'm-^', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', 'm', 'DisplayName', 'WQ + Sum-Fact formation (GPU, FP64)');
    loglog(dofs_all, t_gpu_wq32, 'g-d', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', [0 0.7 0], 'DisplayName', 'WQ + Sum-Fact formation (GPU, FP32)');
    loglog(dofs_all, t_mf64, 'k:v', 'LineWidth', 1.8, 'MarkerSize', 7, 'DisplayName', 'GPU matrix-free apply K(\rho)p (FP64)');
    loglog(dofs_all, t_mf32, 'c:v', 'LineWidth', 1.8, 'MarkerSize', 7, 'DisplayName', 'GPU matrix-free apply K(\rho)p (FP32)');
end
grid on;
set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on', 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.85 0.85 0.85]);
xlabel('Degrees of Freedom (DOFs)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Wall-Clock Time [s]', 'FontSize', 12, 'FontWeight', 'bold');
title('3D Scaling (p = 2): Gauss vs. WQ Formation vs. GPU Operator', 'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'FontSize', 9.5);
xlim([400, 150000]);
exportgraphics(fig, 'fig_timing_h_refinement.png', 'Resolution', 300);
close(fig);
