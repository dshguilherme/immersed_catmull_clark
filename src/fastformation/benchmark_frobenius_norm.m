% BENCHMARK_FROBENIUS_NORM.M
% Comparison 1: Frobenius norm difference of Normal vs Fast Formed assembly.
% Shows that Fast Formation produces the exact same Galerkin matrix up to machine precision.

clearvars; clc; close all;

fprintf('========================================================================\n');
fprintf('  COMPARISON 1: Frobenius Norm Difference (Normal vs Fast Assembly)\n');
fprintf('========================================================================\n\n');

degrees = [2, 3, 4, 5];
nsub_list = {[10, 5], [20, 10], [40, 20]};

results = [];

fprintf('%-8s %-12s %-10s %-16s %-16s %-14s\n', ...
    'Degree p', 'Mesh (nsub)', 'Total DOFs', 'Rel. Frob. Diff', 'Max Abs. Diff', 'Asymmetry Diff');
fprintf('%s\n', repmat('-', 1, 80));

for p = degrees
    for i_m = 1:numel(nsub_list)
        nsub = nsub_list{i_m};
        
        sp = iga_space_box([0 1.0; 0 0.5], nsub, p);

        % 1. Standard Galerkin matrix (exact (p+1)-point Gauss; validated against
        %    GeoPDEs op_su_ev to machine precision in tests/testIgaKernel.m)
        [Ke, type_id] = iga_elasticity_element_matrices(sp, 0.3 / (1.3 * 0.4), 1 / 2.6);
        [r, c] = iga_element_rows_cols(sp.connectivity);
        K_normal = sparse(r, c, reshape(Ke(:, :, type_id), [], 1), sp.ndof, sp.ndof);

        % 2. Fast Formed Assembly (Row Weighted Quadrature + Sum Factorization)
        K_fast = fast_stiffness_assembly(sp, 1.0, 0.3, [], 'element', 3, 1e-9, true);
        
        % Compute difference metrics
        norm_normal = norm(K_normal, 'fro');
        diff_fro = norm(K_fast - K_normal, 'fro') / norm_normal;
        diff_max = full(max(max(abs(K_fast - K_normal))));
        diff_asym = norm(K_fast - K_fast', 'fro') / norm(K_fast, 'fro');
        
        ndof = sp.ndof;
        
        fprintf('%-8d [%2d x %-2d]     %-10d %-16.2e %-16.2e %-14.2e\n', ...
            p, nsub(1), nsub(2), ndof, diff_fro, diff_max, diff_asym);
        
        entry.p = p;
        entry.nsub = nsub;
        entry.ndof = ndof;
        entry.diff_fro = diff_fro;
        entry.diff_max = diff_max;
        entry.diff_asym = diff_asym;
        results = [results; entry];
    end
end

fprintf('%s\n\n', repmat('=', 1, 80));
save('benchmark_frobenius_results.mat', 'results');
