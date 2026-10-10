classdef testIgaKernel < matlab.unittest.TestCase
    % TESTIGAKERNEL
    % Verification of the in-house B-spline / elasticity kernel in src/iga that
    % replaced GeoPDEs. Everything here is checked against exact mathematics; the
    % last test also compares with GeoPDEs when it is available (GEOPDES_PATH).

    properties (Constant)
        lambda = 0.3 / (1.3 * 0.4)   % E = 1, nu = 0.3
        mu = 1 / 2.6
    end

    methods (TestClassSetup)
        function addPaths(~)
            here = fileparts(mfilename('fullpath'));
            addpath(fullfile(here, '..', 'src', 'iga'));
            addpath(fullfile(here, '..', 'src', 'fastformation'));
            addpath(fullfile(here, '..', 'benchmarks'));
        end
    end

    methods (Test)
        function gaussRuleIsExact(testCase)
            for n = 1:8
                [x, w] = iga_gauss_legendre(n);
                for k = 0:2*n-1
                    exact = (1 - (-1)^(k+1)) / (k + 1);
                    testCase.verifyEqual(sum(w .* x.^k), exact, 'AbsTol', 1e-13);
                end
            end
        end

        function basisPartitionOfUnityAndDerivatives(testCase)
            x = linspace(0, 1, 101);
            for p = 1:4
                for reg = [p-1, 0]
                    knots = iga_open_knots(7, p, reg);
                    [N, dN] = iga_bspline_basis(knots, p, x);
                    testCase.verifyEqual(sum(N, 2), ones(numel(x), 1), 'AbsTol', 1e-13);
                    testCase.verifyEqual(sum(dN, 2), zeros(numel(x), 1), 'AbsTol', 1e-10);
                    % central finite differences away from knots (where derivatives may jump)
                    xi = 0.013 + (0:0.07:0.95);
                    h = 1e-6;
                    [~, dNi] = iga_bspline_basis(knots, p, xi);
                    fd = (iga_bspline_basis(knots, p, xi + h) - iga_bspline_basis(knots, p, xi - h)) / (2 * h);
                    testCase.verifyEqual(dNi, fd, 'AbsTol', 1e-6);
                end
            end
        end

        function stiffnessSymmetricWithRigidBodyNullSpace(testCase)
            cases = {[0 1; 0 0.5], [4 3], 3; [0 1.2; 0 0.6; 0 0.3], [3 2 2], 2};
            for c = 1:size(cases, 1)
                sp = iga_space_box(cases{c, 1}, cases{c, 2}, cases{c, 3});
                K = testCase.assemble(sp);
                testCase.verifyLessThan(norm(K - K.', 'fro') / norm(K, 'fro'), 1e-13);
                ev = sort(abs(eig(full(K))));
                nrbm = sp.dim * (sp.dim + 1) / 2;
                testCase.verifyLessThan(max(ev(1:nrbm)) / max(ev), 1e-12);
                testCase.verifyGreaterThan(ev(nrbm + 1) / max(ev), 1e-8);
            end
        end

        function patchTestReproducesLinearField(testCase)
            % Linear displacement imposed at boundary control points must be
            % reproduced exactly in the interior (B-splines reproduce linears
            % with control values at the Greville abscissae).
            cases = {[0 2; -1 0.5], [5 4], 2; [0 1; 0 2; 0.5 1], [3 4 3], 3};
            for c = 1:size(cases, 1)
                sp = iga_space_box(cases{c, 1}, cases{c, 2}, cases{c, 3});
                dim = sp.dim;
                A = reshape(1:dim^2, dim, dim) * 1e-2 - 0.03;
                b = (1:dim).' * 0.1;
                g = cell(1, dim);
                for d = 1:dim
                    t = sp.knots{d}; p = sp.degree(d);
                    gd = arrayfun(@(i) mean(t(i+1:i+p)), 1:sp.ndof_dir(d));
                    g{d} = sp.bounds(d, 1) + sp.L(d) * gd;
                end
                G = cell(1, dim);
                [G{:}] = ndgrid(g{:});
                X = cell2mat(cellfun(@(v) v(:), G, 'UniformOutput', false)); % [ndof_sc x dim]
                u_exact = reshape(X * A.' + b.', [], 1);
                bnd = unique([sp.boundary.dofs]);
                free = setdiff(1:sp.ndof, bnd);
                K = testCase.assemble(sp);
                u = u_exact;
                u(free) = -K(free, free) \ (K(free, bnd) * u_exact(bnd));
                testCase.verifyLessThan(norm(u - u_exact) / norm(u_exact), 1e-11);
            end
        end

        function manufacturedSolutionConvergesAtOptimalRate(testCase)
            % Plane strain on the unit square, u = (sin(pi x) sin(pi y), 0), u = 0 on the boundary.
            lam = testCase.lambda; m = testCase.mu;
            f = @(x, y) cat(1, reshape(pi^2 * (lam + 3*m) * sin(pi*x) .* sin(pi*y), [1 size(x)]), ...
                               reshape(-pi^2 * (lam + m) * cos(pi*x) .* cos(pi*y), [1 size(x)]));
            for p = [2 3]
                nels = [4 8 16];
                err = zeros(size(nels));
                for k = 1:numel(nels)
                    sp = iga_space_box([0 1; 0 1], [nels(k) nels(k)], p);
                    K = testCase.assemble(sp);
                    F = iga_load_vector(sp, f);
                    free = setdiff(1:sp.ndof, unique([sp.boundary.dofs]));
                    u = zeros(sp.ndof, 1);
                    u(free) = K(free, free) \ F(free);
                    [X, W] = iga_quad_points(sp);
                    u1 = iga_eval_scalar_field(sp, u(1:sp.ndof_sc), 'value');
                    u2 = iga_eval_scalar_field(sp, u(sp.ndof_sc+1:end), 'value');
                    e2 = (u1 - sin(pi*X{1}) .* sin(pi*X{2})).^2 + u2.^2;
                    err(k) = sqrt(sum(W(:) .* e2(:)));
                end
                rates = log2(err(1:end-1) ./ err(2:end));
                testCase.verifyGreaterThan(rates(end), p + 1 - 0.25, ...
                    sprintf('p = %d: L2 rates %s', p, mat2str(rates, 3)));
            end
        end

        function weightedQuadratureEqualsGaussForUniformDensity(testCase)
            sp2 = iga_space_box([0 1; 0 0.5], [6 4], 3);
            K2 = testCase.assemble(sp2);
            testCase.verifyLessThan(norm(fast_stiffness_assembly(sp2, 1, 0.3, [], 'element', 3, 1e-3, true) - K2, 'fro') / norm(K2, 'fro'), 1e-12);
            S = wq_setup(sp2, 1, 0.3, 'cpu', 'double');
            testCase.verifyLessThan(norm(wq_form(S, ones(sp2.nel_dir), 'element', 3, 0, true) - K2, 'fro') / norm(K2, 'fro'), 1e-12);
            sp3 = iga_space_box([0 1.2; 0 0.6; 0 0.3], [4 3 2], 2);
            K3 = testCase.assemble(sp3);
            testCase.verifyLessThan(norm(fast_stiffness_assembly(sp3, 1, 0.3, [], 'element', 3, 1e-3, true) - K3, 'fro') / norm(K3, 'fro'), 1e-12);
        end

        function elementSensitivitiesMatchElementEnergies(testCase)
            sp = iga_space_box([0 1; 0 0.5], [6 3], 2);
            rng(3);
            u = randn(sp.ndof, 1);
            x = 0.2 + 0.8 * rand(sp.nel_dir);
            penal = 3; Emin = 1e-3;
            dC = fast_sensitivities(u, sp, x, 'element', penal, Emin, 1, 0.3);
            [Ke, tid] = iga_elasticity_element_matrices(sp, testCase.lambda, testCase.mu);
            ue = u(sp.connectivity);
            Ee = arrayfun(@(e) ue(:, e).' * Ke(:, :, tid(e)) * ue(:, e), 1:sp.nel);
            ref = -penal * x(:).^(penal - 1) * (1 - Emin) .* Ee(:);
            testCase.verifyEqual(dC(:), ref, 'RelTol', 1e-11);
        end

        function matchesGeoPDEsWhenAvailable(testCase)
            testCase.assumeTrue(geopdes_baseline_available(), 'GeoPDEs not available (set GEOPDES_PATH).');
            for p = 1:3
                bounds = [0.5 2.5; -1 0.2; 0.1 0.4];
                [~, Kg] = geopdes_gauss_baseline(bounds, [4 3 3], p, testCase.lambda, testCase.mu);
                K = testCase.assemble(iga_space_box(bounds, [4 3 3], p));
                testCase.verifyLessThan(norm(K - Kg, 'fro') / norm(Kg, 'fro'), 1e-13);
            end
        end
    end

    methods (Static)
        function K = assemble(sp)
            [Ke, tid] = iga_elasticity_element_matrices(sp, testIgaKernel.lambda, testIgaKernel.mu);
            [r, c] = iga_element_rows_cols(sp.connectivity);
            K = sparse(r, c, reshape(Ke(:, :, tid), [], 1), sp.ndof, sp.ndof);
        end
    end
end
