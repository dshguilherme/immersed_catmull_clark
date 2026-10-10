% RUN_PAPER_TOPOPT_2D  2D cantilever topology optimization of the FastFormation paper
% (Section "High-Resolution Isogeometric Topology Optimization"), run through the
% weighted-quadrature pipeline (WQ_SETUP once, WQ_FORM + FAST_SENSITIVITIES per
% iteration) with the corrected tip load (IGA_CANTILEVER_TIP_LOAD).
%   80 x 40 elements, p = 3 (C^2), f0 = 0.5, p_simp = 3, r_min = 2, 50 iterations.
%   Option A: element densities + sensitivity filter.
%   Option B: control-point densities, evaluated at the WQ points (no filter).
% Writes topopt_wq_2d_results.mat and fig_topopt_element_p3.png / fig_topopt_spline_p3.png.

clearvars; close all;
nel = [80 40]; volfrac = 0.5; penal = 3; rmin = 2.0; max_iter = 50; p = 3;
opts = struct('layout', 'interior', 'device', 'cpu', 'tol', 0, 'verbose', true);

R = struct();
for opt = {'element', 'spline'}
    fprintf('\n=== Option %s ===\n', opt{1});
    t0 = tic;
    [x, c, t, info] = topopt_iga_wq(nel, volfrac, penal, rmin, max_iter, opt{1}, p, opts);
    R.(opt{1}) = struct('x', x, 'c', c, 't', t, 'change', info.change, 'rho_elem', info.rho_elem, ...
        't_assembly', info.t_assembly, 't_setup', info.t_setup, 't_total', toc(t0), 'ndof', info.space.ndof);
    fprintf('Option %s: J0 = %.4f, J(%d) = %.4f (%.1f%% reduction), total %.2f s (%.3f s/iter, assembly %.3f s/iter)\n', ...
        opt{1}, c(1), numel(c), c(end), 100 * (1 - c(end) / c(1)), sum(t), mean(t), mean(info.t_assembly));
end
save('topopt_wq_2d_results.mat', 'R', 'nel', 'volfrac', 'penal', 'rmin', 'max_iter', 'p');

%% Figures (same layout as the paper)
L = 1.0; h = 0.5;
for opt = {'element', 'spline'}
    r = R.(opt{1});
    fig = figure('Position', [100, 100, 1000, 420], 'Color', 'w', 'InvertHardcopy', 'off', 'Visible', 'off');
    subplot(1, 2, 1);
    imagesc([0 L], [0 h], r.rho_elem');
    colormap(flipud(gray)); axis equal tight;
    set(gca, 'YDir', 'normal', 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'Box', 'on', 'LineWidth', 1.2);
    clim([0 1]); cb = colorbar; set(cb, 'Color', 'k', 'FontSize', 10);
    ylabel(cb, 'Physical Density \rho', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
    title(sprintf('Optimal Topology (%s, p = %d, %dx%d)', opt{1}, p, nel(1), nel(2)), 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    xlabel('x / L', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('y / h', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
    subplot(1, 2, 2);
    n = numel(r.c);
    yyaxis left;
    plot(1:n, r.c, 'b-o', 'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'b');
    ylabel('Compliance J = F^T U', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'b'); set(gca, 'YColor', 'b');
    yyaxis right;
    semilogy(1:n, r.change, 'r--s', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'r');
    ylabel('Max Change \Delta \rho_{max}', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r'); set(gca, 'YColor', 'r');
    grid on;
    set(gca, 'FontSize', 11, 'XColor', 'k', 'Color', 'w', 'Box', 'on', 'GridColor', [0.85 0.85 0.85], 'LineWidth', 1.2);
    title('Convergence History (Compliance & Change)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    xlabel('Iteration', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
    exportgraphics(fig, sprintf('fig_topopt_%s_p%d.png', opt{1}, p), 'Resolution', 200);
    close(fig);
end
