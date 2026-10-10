% BENCHMARK_FROBENIUS_NORM.M
% Comparison 1: Frobenius norm difference of Normal vs Fast Formed assembly.
%   (a) uniform density: WQ (interior layout) reproduces the Gauss Galerkin matrix
%       to machine precision; compared both with the exact (p+1)-point Gauss
%       matrix of IGA_ELASTICITY_ELEMENT_MATRICES and, when available, with
%       GeoPDEs op_su_ev_tp (GEOPDES_PATH).
%   (b) random element-wise density: WQ is a quadrature approximation of the
%       Gauss matrix with the same piecewise-constant coefficient; the relative
%       difference is reported for the interior and the Calabro layouts.

clearvars; clc; close all;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'benchmarks'));
has_geopdes = geopdes_baseline_available();

fprintf('========================================================================\n');
fprintf('  COMPARISON 1: Frobenius Norm Difference (Normal vs Fast Assembly)\n');
fprintf('========================================================================\n\n');

degrees = [2, 3, 4, 5];
nsub_list = {[10, 5], [20, 10], [40, 20]};
lambda = 0.3 / (1.3 * 0.4); mu = 1 / 2.6;
penal = 3; Emin = 1e-3;
rng(42);

results = [];
fprintf('%-4s %-9s %-6s | %-10s %-10s %-10s %-10s | %-11s %-11s\n', 'p', 'Mesh', 'DOFs', ...
    'E_F', 'E_max', 'E_F(GeoP)', 'asym', 'E_F rho int', 'E_F rho Cal');
fprintf('%s\n', repmat('-', 1, 96));

for p = degrees
    for i_m = 1:numel(nsub_list)
        nsub = nsub_list{i_m};
        sp = iga_space_box([0 1.0; 0 0.5], nsub, p);

        % Standard Galerkin matrix: exact (p+1)-point Gauss
        [Ke, type_id] = iga_elasticity_element_matrices(sp, lambda, mu);
        [r, c] = iga_element_rows_cols(sp.connectivity);
        K_normal = sparse(r, c, reshape(Ke(:, :, type_id), [], 1), sp.ndof, sp.ndof);

        % (a) uniform density
        K_fast = fast_stiffness_assembly(sp, 1.0, 0.3, [], 'element', penal, Emin, true, sp, 'interior');
        e_fro = norm(K_fast - K_normal, 'fro') / norm(K_normal, 'fro');
        e_max = full(max(max(abs(K_fast - K_normal))));
        e_asym = norm(K_fast - K_fast', 'fro') / norm(K_fast, 'fro');
        e_geo = NaN;
        if has_geopdes
            [~, K_geo] = geopdes_gauss_baseline([0 1.0; 0 0.5], nsub, p, lambda, mu);
            e_geo = norm(K_fast - K_geo, 'fro') / norm(K_geo, 'fro');
        end

        % (b) random element-wise density, same coefficient in Gauss and WQ
        x = rand(nsub);
        s = Emin + (1 - Emin) * x(:).^penal;
        K_rho = sparse(r, c, reshape(Ke(:, :, type_id) .* reshape(s, 1, 1, []), [], 1), sp.ndof, sp.ndof);
        K_int = fast_stiffness_assembly(sp, 1.0, 0.3, x, 'element', penal, Emin, true, sp, 'interior');
        K_cal = fast_stiffness_assembly(sp, 1.0, 0.3, x, 'element', penal, Emin, true, sp, 'calabro');
        e_rho_int = norm(K_int - K_rho, 'fro') / norm(K_rho, 'fro');
        e_rho_cal = norm(K_cal - K_rho, 'fro') / norm(K_rho, 'fro');

        fprintf('%-4d %2dx%-6d %-6d | %-10.2e %-10.2e %-10.2e %-10.2e | %-11.2e %-11.2e\n', ...
            p, nsub(1), nsub(2), sp.ndof, e_fro, e_max, e_geo, e_asym, e_rho_int, e_rho_cal);

        entry = struct('p', p, 'nsub', nsub, 'ndof', sp.ndof, 'diff_fro', e_fro, 'diff_max', e_max, ...
            'diff_fro_geopdes', e_geo, 'diff_asym', e_asym, 'diff_rho_interior', e_rho_int, 'diff_rho_calabro', e_rho_cal);
        results = [results; entry]; %#ok<AGROW>
    end
end

fprintf('%s\n\n', repmat('=', 1, 96));
save('benchmark_frobenius_results.mat', 'results');
