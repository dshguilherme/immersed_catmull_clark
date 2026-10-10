% BENCHMARK_WQ_CONSISTENCY
% Supports / tests Proposition 1(4) of the FastFormation paper: convergence of the
% WQ-formed Galerkin solution to the Gauss-formed one for a discontinuous
% (piecewise-constant) SIMP coefficient under h-refinement.
%   Cantilever [0,1]x[0,0.5], clamped at x = 0, unit tip load (IGA_CANTILEVER_TIP_LOAD),
%   density piecewise constant on a fixed 4 x 2 block pattern (aligned with every
%   mesh), SIMP p = 3. For each mesh: u_G (exact Gauss), u_WQ (interior and Calabro
%   layouts), and the reference u_ref from Gauss on the finest mesh + 1 level.
%   Reports the relative energy-norm difference ||u_WQ - u_G||_{K_G} / ||u_G||_{K_G},
%   the relative compliance difference |J_WQ - J_G| / J_G, and the discretization
%   error of the Gauss solution |J_G - J_ref| / J_ref, with observed rates.

clearvars; clc;
lambda = 0.3 / (1.3 * 0.4); mu = 1 / 2.6; penal = 3; Emin = 1e-3;
blocks = [0.9 0.2; 0.3 1.0; 1.0 0.5; 0.15 0.8];      % 4 x 2 block densities
levels = 0:4;
R = struct();
for p = [2 3]
    nl = numel(levels) + 1;
    J = nan(nl, 3); dE = nan(nl, 2); hh = nan(nl, 1);
    for k = 1:nl
        nel = [8 4] * 2^(k - 1);
        hh(k) = 1 / nel(1);
        sp = iga_space_box([0 1; 0 0.5], nel, p);
        x = kron(blocks, ones(nel ./ [4 2]));          % element densities on the block pattern
        free = setdiff(1:sp.ndof, sp.boundary(1).dofs);
        F = iga_load_vector(sp, @(xx, yy) iga_cantilever_tip_load(xx, yy, 1, 0.5, 'center'));
        F = F / abs(sum(F(sp.ndof_sc + 1:end)));
        [Ke, tid] = iga_elasticity_element_matrices(sp, lambda, mu);
        [r, c] = iga_element_rows_cols(sp.connectivity);
        s = Emin + (1 - Emin) * x(:).^penal;
        KG = sparse(r, c, reshape(Ke(:, :, tid) .* reshape(s, 1, 1, []), [], 1), sp.ndof, sp.ndof);
        uG = zeros(sp.ndof, 1); uG(free) = KG(free, free) \ F(free);
        J(k, 1) = F' * uG;
        if k == nl, break; end                            % finest level: reference only
        lay = {'interior', 'calabro'};
        for m = 1:2
            KW = wq_form(wq_setup(sp, 1, 0.3, 'cpu', 'double', lay{m}), x, 'element', penal, Emin, true);
            uW = zeros(sp.ndof, 1); uW(free) = KW(free, free) \ F(free);
            J(k, m + 1) = F' * uW;
            d = uW - uG;
            dE(k, m) = sqrt(d' * KG * d) / sqrt(uG' * KG * uG);
        end
        fprintf('p %d  %3dx%-3d  J_G %.6f  |J_int-J_G|/J_G %.2e  |J_cal-J_G|/J_G %.2e  E_int %.2e  E_cal %.2e\n', ...
            p, nel, J(k, 1), abs(J(k, 2) - J(k, 1)) / J(k, 1), abs(J(k, 3) - J(k, 1)) / J(k, 1), dE(k, 1), dE(k, 2));
    end
    Jref = J(nl, 1);
    nm = nl - 1;
    eJG = abs(J(1:nm, 1) - Jref) / Jref;
    eJW = abs(J(1:nm, 2) - J(1:nm, 1)) ./ J(1:nm, 1);
    rate = @(e) log(e(1:end-1) ./ e(2:end)) ./ log(hh(1:nm-1) ./ hh(2:nm));
    fprintf('p %d rates: Gauss discretization (compliance) %s\n', p, mat2str(rate(eJG).', 3));
    fprintf('p %d rates: WQ interior consistency, compliance %s, energy %s\n', p, mat2str(rate(eJW).', 3), mat2str(rate(dE(1:nm, 1)).', 3));
    fprintf('p %d rates: WQ Calabro  consistency, energy %s\n\n', p, mat2str(rate(dE(1:nm, 2)).', 3));
    R.(sprintf('p%d', p)) = struct('h', hh(1:nm), 'J', J, 'dE', dE(1:nm, :), 'eJG', eJG, 'eJW', eJW);
end
save('benchmark_wq_consistency.mat', 'R', 'blocks', 'levels');
