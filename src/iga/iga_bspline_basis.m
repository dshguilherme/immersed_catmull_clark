function [N, dN] = iga_bspline_basis(knots, degree, x)
% IGA_BSPLINE_BASIS  Values and first derivatives of all B-splines at points x.
%   [N, dN] = iga_bspline_basis(knots, degree, x) returns dense matrices of size
%   [numel(x) x ndof], ndof = numel(knots) - degree - 1. Derivatives are with
%   respect to the knot-vector parameter. Points at the right end of the knot
%   vector are assigned to the last non-empty span (right-continuous basis
%   except at the end point).

knots = knots(:).';
x = x(:);
p = degree;
ndof = numel(knots) - p - 1;
nx = numel(x);
N = zeros(nx, ndof);
dN = zeros(nx, ndof);

% Span index s (1-based): knots(s) <= x < knots(s+1), clamped to the valid range.
span = sum(knots <= x, 2);
span = min(max(span, p + 1), ndof);

for q = 1:nx
    s = span(q);
    xq = x(q);
    b = 1;            % degree-0 basis on the span
    b_prev = [];
    for k = 1:p
        if k == p, b_prev = b; end
        bn = zeros(1, k + 1);
        for r = 0:k
            i = s - k + r;            % global index of the degree-k function
            val = 0;
            if r > 0
                d = knots(i + k) - knots(i);
                if d > 0, val = val + (xq - knots(i)) / d * b(r); end
            end
            if r < k
                d = knots(i + k + 1) - knots(i + 1);
                if d > 0, val = val + (knots(i + k + 1) - xq) / d * b(r + 1); end
            end
            bn(r + 1) = val;
        end
        b = bn;
    end
    idx = (s - p):s;
    N(q, idx) = b;
    if p > 0
        db = zeros(1, p + 1);
        for r = 0:p
            i = s - p + r;
            val = 0;
            if r > 0
                d = knots(i + p) - knots(i);
                if d > 0, val = val + b_prev(r) / d; end
            end
            if r < p
                d = knots(i + p + 1) - knots(i + 1);
                if d > 0, val = val - b_prev(r + 1) / d; end
            end
            db(r + 1) = p * val;
        end
        dN(q, idx) = db;
    end
end
end
