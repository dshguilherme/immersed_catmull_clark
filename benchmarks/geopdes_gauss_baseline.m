function [t, K] = geopdes_gauss_baseline(bounds, nsub, degree, lambda, mu, xPhys, penal, Emin)
% GEOPDES_GAUSS_BASELINE  Time GeoPDEs' standard Gauss assembly (op_su_ev_tp) on a box.
%   [t, K] = geopdes_gauss_baseline(bounds, nsub, degree, lambda, mu) returns the
%   wall-clock time of the tensor-product Gauss assembly in GeoPDEs and the
%   matrix, with (p+1)^d Gauss points per element.
%   [t, K] = geopdes_gauss_baseline(..., xPhys, penal, Emin) assembles with the
%   SIMP-scaled Lame coefficients lambda*s(x), mu*s(x), s = Emin + (1-Emin) rho^penal,
%   where rho is the element-wise density xPhys (size nsub) looked up at every
%   Gauss point (the same heterogeneous operator the WQ benchmarks form).
%   Returns t = NaN, K = [] when GeoPDEs is not available (see
%   GEOPDES_BASELINE_AVAILABLE). Used only as an external reference in timing
%   benchmarks; nothing in src/ depends on it.

t = NaN; K = [];
if ~geopdes_baseline_available()
    return;
end
if nargin < 6, xPhys = []; end
if nargin < 7 || isempty(penal), penal = 3; end
if nargin < 8 || isempty(Emin), Emin = 1e-3; end
dim = size(bounds, 1);
lo = bounds(:, 1).'; L = (bounds(:, 2) - bounds(:, 1)).';
if isempty(xPhys)
    sfun = @(varargin) ones(size(varargin{1}));
else
    s = Emin + (1 - Emin) * xPhys.^penal;
    h = L ./ nsub;
    idx = @(x, d) min(nsub(d), max(1, floor((x - lo(d)) / h(d)) + 1));
    if dim == 2
        sfun = @(x, y) reshape(s(sub2ind(nsub, idx(x(:), 1), idx(y(:), 2))), size(x));
    else
        sfun = @(x, y, z) reshape(s(sub2ind(nsub, idx(x(:), 1), idx(y(:), 2), idx(z(:), 3))), size(x));
    end
end
if dim == 2
    pdata.geo_name = nrb4surf(lo, lo + [L(1) 0], lo + [0 L(2)], lo + L);
    pdata.lambda_lame = @(x, y) lambda * sfun(x, y);
    pdata.mu_lame = @(x, y) mu * sfun(x, y);
else
    srf = nrb4surf(lo, lo + [L(1) 0 0], lo + [0 L(2) 0], lo + [L(1) L(2) 0]);
    pdata.geo_name = nrbextrude(srf, [0 0 L(3)]);
    pdata.lambda_lame = @(x, y, z) lambda * sfun(x, y, z);
    pdata.mu_lame = @(x, y, z) mu * sfun(x, y, z);
end
pdata.drchlt_sides = []; pdata.nmnn_sides = []; pdata.press_sides = []; pdata.symm_sides = [];
mdata.degree = degree * ones(1, dim);
mdata.regularity = (degree - 1) * ones(1, dim);
mdata.nsub = nsub;
mdata.nquad = (degree + 1) * ones(1, dim);
[~, msh, sp] = buildSpaces(pdata, mdata);
t0 = tic;
K = op_su_ev_tp(sp, sp, msh, pdata.lambda_lame, pdata.mu_lame);
t = toc(t0);
end
