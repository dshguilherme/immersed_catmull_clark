% RUN_ALL_COMPARISONS.M
% Master execution script for all 4 comparisons requested:
% 1. Frobenius norm difference (Normal vs Fast Formed assembly)
% 2. Assembly timing vs h-refinement (Normal vs Fast CPU vs GPU)
% 3. Assembly timing vs p-refinement (C^0 continuity)
% 4. Assembly timing vs k-refinement (C^{p-1} maximal continuity)

clearvars; clc; close all;
addpath(genpath('C:\Users\dshgu\OneDrive\Documents\geopdes-master'));
addpath('c:\Users\dshgu\OneDrive\Documents\FastFormation');

fprintf('########################################################################\n');
fprintf('  STARTING COMPLETE BENCHMARK SUITE FOR FAST MATRIX FORMATION IN IGA   \n');
fprintf('########################################################################\n\n');

% Run Comparison 1
fprintf('>>> RUNNING COMPARISON 1: Frobenius Norm Verification...\n');
run('benchmark_frobenius_norm.m');

% Run Comparison 2
fprintf('\n>>> RUNNING COMPARISON 2: Assembly Timing vs h-Refinement...\n');
run('benchmark_h_refinement.m');

% Run Comparison 3
fprintf('\n>>> RUNNING COMPARISON 3: Assembly Timing vs p-Refinement (C^0)...\n');
run('benchmark_p_refinement.m');

% Run Comparison 4
fprintf('\n>>> RUNNING COMPARISON 4: Assembly Timing vs k-Refinement (C^{p-1})...\n');
run('benchmark_k_refinement.m');

fprintf('\n########################################################################\n');
fprintf('  ALL 4 BENCHMARKS SUCCESSFULLY COMPLETED!                             \n');
fprintf('  Generated figures:                                                    \n');
fprintf('    - fig_timing_h_refinement.png                                      \n');
fprintf('    - fig_timing_p_refinement.png                                      \n');
fprintf('    - fig_timing_k_refinement.png                                      \n');
fprintf('  Generated data files:                                                \n');
fprintf('    - benchmark_frobenius_results.mat                                  \n');
fprintf('    - benchmark_h_results.mat                                          \n');
fprintf('    - benchmark_p_results.mat                                          \n');
fprintf('    - benchmark_k_results.mat                                          \n');
fprintf('########################################################################\n');
