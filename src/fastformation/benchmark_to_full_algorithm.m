% benchmark_to_full_algorithm.m
% Complete End-to-End Topology Optimization Benchmark Suite:
% Comparing CPU WQ Direct vs GPU Matrix-Free (FP64 & FP32)
% in both 2D and 3D settings.

clearvars; clc; close all;
addpath(genpath('C:\Users\dshgu\OneDrive\Documents\geopdes-master'));
addpath('c:\Users\dshgu\OneDrive\Documents\FastFormation');

fprintf('========================================================================\n');
fprintf('  BENCHMARK: COMPLETE END-TO-END TOPOLOGY OPTIMIZATION ALGORITHMS\n');
fprintf('========================================================================\n\n');

%% 1. 2D End-to-End Benchmark (80 x 40 elements, 7138 DOFs, p = 3)
fprintf('>>> 1. Benchmarking 2D Full Topology Optimization (30 iterations)...\n');
n_iter_2d = 30;

% 1.1 CPU WQ Direct Solver
fprintf('  -> Running 2D CPU WQ Direct...\n');
tic;
[x_cpu, c_cpu, t_cpu] = topopt_iga_fast(80, 40, 0.5, 3.0, 2.0, n_iter_2d, 'element', 3);
time_2d_cpu = toc;

% 1.2 GPU Matrix-Free FP64
fprintf('  -> Running 2D GPU Matrix-Free (FP64)...\n');
tic;
[x_g64, c_g64, t_g64] = topopt_iga_2d_mf(80, 40, 0.5, 3.0, 2.0, n_iter_2d, 3, 'fp64');
time_2d_g64 = toc;

% 1.3 GPU Matrix-Free FP32
fprintf('  -> Running 2D GPU Matrix-Free (FP32)...\n');
tic;
[x_g32, c_g32, t_g32] = topopt_iga_2d_mf(80, 40, 0.5, 3.0, 2.0, n_iter_2d, 3, 'fp32');
time_2d_g32 = toc;

%% 2. 3D End-to-End Benchmark (16 x 8 x 4 elements, 3240 DOFs, p = 2)
fprintf('\n>>> 2. Benchmarking 3D Full Topology Optimization (25 iterations)...\n');
n_iter_3d = 25;

% 2.1 CPU Direct Solver
fprintf('  -> Running 3D CPU Direct...\n');
[~, c_3d_cpu, t_3d_cpu, time_3d_cpu] = topopt_iga_3d(16, 8, 4, 0.3, 3.0, 1.5, n_iter_3d, 'cpu');

% 2.2 GPU Matrix-Free FP64
fprintf('  -> Running 3D GPU Matrix-Free (FP64)...\n');
[~, c_3d_g64, t_3d_g64, time_3d_g64] = topopt_iga_3d(16, 8, 4, 0.3, 3.0, 1.5, n_iter_3d, 'gpu_mf_fp64');

% 2.3 GPU Matrix-Free FP32
fprintf('  -> Running 3D GPU Matrix-Free (FP32)...\n');
[~, c_3d_g32, t_3d_g32, time_3d_g32] = topopt_iga_3d(16, 8, 4, 0.3, 3.0, 1.5, n_iter_3d, 'gpu_mf_fp32');

%% 3. Summary & Comparison Table
fprintf('\n========================================================================\n');
fprintf('  SUMMARY RESULTS: END-TO-END RUNTIMES\n');
fprintf('========================================================================\n');
fprintf('2D Benchmark (80x40 elements, 7,138 DOFs, %d iterations):\n', n_iter_2d);
fprintf('  CPU WQ Direct        : %7.2f s (avg: %6.3f s/iter)\n', time_2d_cpu, mean(t_cpu));
fprintf('  GPU Matrix-Free FP64 : %7.2f s (avg: %6.3f s/iter)\n', time_2d_g64, mean(t_g64));
fprintf('  GPU Matrix-Free FP32 : %7.2f s (avg: %6.3f s/iter) -> Speedup vs CPU: %.2fx\n', ...
    time_2d_g32, mean(t_g32), time_2d_cpu / time_2d_g32);

fprintf('\n3D Benchmark (16x8x4 elements, 3,240 DOFs, %d iterations):\n', n_iter_3d);
fprintf('  CPU Direct           : %7.2f s (avg: %6.3f s/iter)\n', time_3d_cpu, mean(t_3d_cpu));
fprintf('  GPU Matrix-Free FP64 : %7.2f s (avg: %6.3f s/iter)\n', time_3d_g64, mean(t_3d_g64));
fprintf('  GPU Matrix-Free FP32 : %7.2f s (avg: %6.3f s/iter) -> Speedup vs CPU: %.2fx\n', ...
    time_3d_g32, mean(t_3d_g32), time_3d_cpu / time_3d_g32);

save('benchmark_full_to_results.mat', ...
    'time_2d_cpu', 'time_2d_g64', 'time_2d_g32', 't_cpu', 't_g64', 't_g32', ...
    'time_3d_cpu', 'time_3d_g64', 'time_3d_g32', 't_3d_cpu', 't_3d_g64', 't_3d_g32');
