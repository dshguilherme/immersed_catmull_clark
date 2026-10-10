function [xPhys, c_hist, t_hist, info] = topopt_iga_wq(nel, volfrac, penal, rmin, max_iter, density_type, degree, opts)
% TOPOPT_IGA_WQ  Isogeometric SIMP topology optimization with weighted-quadrature
% stiffness formation and fast quadrature-point sensitivities (FastFormation paper,
% Algorithm 1, CPU path).
%
%   nel          - [nelx nely] (2D cantilever [0,1]x[0,0.5]) or [nelx nely nelz]
%                  (3D cantilever [0,1.2]x[0,0.6]x[0,0.3])
%   density_type - 'element' (Option A: element densities + sensitivity filter) or
%                  'spline'  (Option B: control-point densities in the displacement
%                             space, evaluated at the WQ points; no filter)
%   opts         - .layout ('interior' | 'calabro'), .device ('cpu' | 'gpu'),
%                  .tol (default 1e-3), .move (0.2), .verbose (true)
%
%   Setup once (WQ_SETUP); per iteration: K = WQ_FORM(rho) (symmetrized), U = K \ F, dC from the strain energy at
%   Gauss points (FAST_SENSITIVITIES), OC update with bisection initialised in the
%   active sensitivity range. 2D load: unit downward traction patch at the centre of
%   the free end (IGA_CANTILEVER_TIP_LOAD, normalized). 3D load and clamp as TOPOPT_IGA_3D.

if nargin < 8, opts = struct(); end
if ~isfield(opts, 'layout'), opts.layout = 'interior'; end
if ~isfield(opts, 'device'), opts.device = 'cpu'; end
if ~isfield(opts, 'tol'), opts.tol = 1e-3; end
if ~isfield(opts, 'move'), opts.move = 0.2; end
if ~isfield(opts, 'verbose'), opts.verbose = true; end
E0 = 1.0; nu = 0.3; Emin = 1e-3;
dim = numel(nel);
if dim == 2
    bounds = [0 1; 0 0.5];
else
    bounds = [0 1.2; 0 0.6; 0 0.3];
end
sp = iga_space_box(bounds, nel, degree);

% Boundary conditions and load
free = setdiff(1:sp.ndof, sp.boundary(1).dofs);
if dim == 2
    F = iga_load_vector(sp, @(x, y) iga_cantilever_tip_load(x, y, bounds(1,2), bounds(2,2), 'center'));
    F = F / abs(sum(F(sp.ndof_sc + 1:2 * sp.ndof_sc)));
else
    ncp = sp.ndof_dir;
    [i1, i2, i3] = ind2sub(ncp, 1:sp.ndof_sc);
    load_sc = find(i1 == ncp(1) & abs(i2 - 1) <= 1 & abs(i3 - ncp(3) / 2) <= 1);
    F = zeros(sp.ndof, 1);
    F(load_sc + sp.ndof_sc) = -1 / numel(load_sc);
end

% Design variables, volume weights and filter
spline = strcmpi(density_type, 'spline');
if spline
    x = volfrac * ones(sp.ndof_dir);
    dV = reshape(iga_integrate_scalar(sp, ones(sp.nqn, sp.nel)), size(x));   % int N_A
else
    x = volfrac * ones(nel);
    h = sp.L ./ nel;
    dV = prod(h) * ones(nel);
    sub = cell(1, dim); [sub{:}] = ind2sub(nel, (1:sp.nel).');
    C = cell2mat(sub);
    [ii, jj, dd] = deal([]);
    for e = 1:sp.nel
        d = sqrt(sum((C - C(e, :)).^2, 2));
        nb = find(d < rmin);
        ii = [ii; e * ones(numel(nb), 1)]; jj = [jj; nb]; dd = [dd; rmin - d(nb)]; %#ok<AGROW>
    end
    H = sparse(ii, jj, dd, sp.nel, sp.nel);
    Hs = full(sum(H, 2));
end
V0 = prod(sp.L);
S = wq_setup(sp, E0, nu, opts.device, 'double', opts.layout);
t_assembly = zeros(max_iter, 1);

c_hist = zeros(max_iter, 1); t_hist = zeros(max_iter, 1); ch_hist = zeros(max_iter, 1);
change = 1; iter = 0;
while iter < max_iter && change > opts.tol
    iter = iter + 1;
    t0 = tic;
    ta = tic;
    K = wq_form(S, x, density_type, penal, Emin, true, 'cpu_sparse');
    t_assembly(iter) = toc(ta);
    U = zeros(sp.ndof, 1);
    U(free) = K(free, free) \ F(free);
    c = F.' * U;
    dC = fast_sensitivities(U, sp, x, density_type, penal, Emin, E0, nu);
    if ~spline
        dC = reshape((H * (x(:) .* dC(:))) ./ (Hs .* max(1e-3, x(:))), size(x));
    end
    % OC with bisection in the active sensitivity range
    sens = max(0, -dC ./ dV);
    l1 = 0; l2 = max(sens(:));
    xold = x;
    upd = @(l) max(1e-3, max(xold - opts.move, min(1, min(xold + opts.move, xold .* sqrt(sens / l)))));
    while sum(dV(:) .* reshape(upd(l2), [], 1)) > volfrac * V0
        l2 = 2 * l2;
    end
    while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-6
        lm = 0.5 * (l1 + l2);
        if sum(dV(:) .* reshape(upd(lm), [], 1)) > volfrac * V0, l1 = lm; else, l2 = lm; end
    end
    x = upd(l2);
    change = max(abs(x(:) - xold(:)));
    c_hist(iter) = c; t_hist(iter) = toc(t0); ch_hist(iter) = change;
    if opts.verbose
        fprintf('%4d  J = %12.6f  vol = %.4f  change = %.4e  t = %.3f s\n', iter, c, sum(dV(:) .* x(:)) / V0, change, t_hist(iter));
    end
end
c_hist = c_hist(1:iter); t_hist = t_hist(1:iter);
info.change = ch_hist(1:iter); info.t_assembly = t_assembly(1:iter); info.t_setup = S.t_setup; info.space = sp; info.dV = dV;
if spline
    % physical density at element centres for plotting
    cen = cell(1, dim);
    for d = 1:dim
        cen{d} = iga_bspline_basis(sp.knots{d}, degree, 0.5 * (sp.breaks{d}(1:end-1) + sp.breaks{d}(2:end)));
    end
    if dim == 2
        info.rho_elem = cen{1} * x * cen{2}.';
    else
        info.rho_elem = [];
    end
else
    info.rho_elem = x;
end
xPhys = x;
end
