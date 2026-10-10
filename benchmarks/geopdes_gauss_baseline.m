function [t, K] = geopdes_gauss_baseline(bounds, nsub, degree, lambda, mu)
% GEOPDES_GAUSS_BASELINE  Time GeoPDEs' standard Gauss assembly (op_su_ev_tp) on a box.
%   [t, K] = geopdes_gauss_baseline(bounds, nsub, degree, lambda, mu) returns the
%   wall-clock time of the tensor-product Gauss assembly in GeoPDEs and the
%   matrix. Returns t = NaN, K = [] when GeoPDEs is not available (see
%   GEOPDES_BASELINE_AVAILABLE). Used only as an external reference in timing
%   benchmarks; nothing in src/ depends on it.

t = NaN; K = [];
if ~geopdes_baseline_available()
    return;
end
dim = size(bounds, 1);
lo = bounds(:, 1).'; L = (bounds(:, 2) - bounds(:, 1)).';
if dim == 2
    pdata.geo_name = nrb4surf(lo, lo + [L(1) 0], lo + [0 L(2)], lo + L);
    pdata.lambda_lame = @(x, y) lambda * ones(size(x));
    pdata.mu_lame = @(x, y) mu * ones(size(x));
else
    srf = nrb4surf(lo, lo + [L(1) 0 0], lo + [0 L(2) 0], lo + [L(1) L(2) 0]);
    pdata.geo_name = nrbextrude(srf, [0 0 L(3)]);
    pdata.lambda_lame = @(x, y, z) lambda * ones(size(x));
    pdata.mu_lame = @(x, y, z) mu * ones(size(x));
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
