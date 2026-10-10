% RUN_NIST_IMMERSED_TOPOPT
% Runs 3D Immersed Topology Optimization on the NIST STEP PMI Benchmark Model (CTC-01)
% using Catmull-Clark / uniform B-splines, cut-cell Weighted Quadrature,
% Ghost Penalty stabilization, and FastFormation GPU Matrix-Free PCG solver.
clear; clc; close all;
addpath(genpath('src'));

step_file = 'C:/Users/dshgu/immersed-iga/Models/NIST-PMI-STEP-Files/NIST-PMI-STEP-Files/AP203 geometry only/nist_ctc_01_asme1_rd.stp';

opts.grid_res = [24, 18, 14];
opts.volfrac = 0.35;
opts.max_iter = 25;
opts.penal = 3.0;
opts.rmin = 1.8;
opts.gamma_gp = 0.05;
opts.pcg_tol = 1e-4;
opts.pcg_maxit = 120;
opts.output_fig = fullfile(pwd, 'figures', 'fig_topopt_nist_ctc01_immersed.png');

results = topopt_immersed_iga_3d(step_file, opts);

% Save numerical history
save(fullfile(pwd, 'figures', 'nist_topopt_results.mat'), 'results');
fprintf('Results saved to figures/nist_topopt_results.mat\n');
