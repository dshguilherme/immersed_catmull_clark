% test_topopt_run.m

fprintf('========================================================================\n');
fprintf('  RUNNING HIGH-RESOLUTION IGA TOPOLOGY OPTIMIZATION EXPERIMENTS\n');
fprintf('========================================================================\n\n');

fprintf('>>> 1. Testing Option A: Element-Wise Density (80 x 40 elements, p = 3, 50 iterations)...\n');
[xPhys_elem, c_hist_elem, t_hist_elem] = topopt_iga_fast(80, 40, 0.5, 3.0, 2.0, 50, 'element', 3);

fprintf('\n>>> 2. Testing Option B: Continuous B-Spline Density (80 x 40 elements, p = 3, 50 iterations)...\n');
[xPhys_spline, c_hist_spline, t_hist_spline] = topopt_iga_fast(80, 40, 0.5, 3.0, 2.0, 50, 'spline', 3);

save('topopt_results.mat', 'xPhys_elem', 'c_hist_elem', 't_hist_elem', 'xPhys_spline', 'c_hist_spline', 't_hist_spline');
fprintf('\nAll high-resolution topology optimization experiments completed successfully!\n');
