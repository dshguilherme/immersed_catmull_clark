function s = iga_space_1d(knots, degree, nquad)
% IGA_SPACE_1D  Univariate B-spline space on [knots(1), knots(end)] with Gauss data.
%   s = iga_space_1d(knots, degree, nquad) returns a struct with fields
%     knots, degree, ndof, nel, breaks
%     connectivity  [(p+1) x nel]  global indices of the functions nonzero on each element
%     supp          {1 x ndof}     elements in the support of each function
%     qn, qw        [nquad x nel]  Gauss nodes and weights (parametric)
%     shape_functions, shape_function_gradients  [nquad x (p+1) x nel]
%   Derivatives are parametric (with respect to the knot parameter).

knots = knots(:).';
p = degree;
breaks = unique(knots);
nel = numel(breaks) - 1;
ndof = numel(knots) - p - 1;

% Functions nonzero on each element: span of the element midpoint.
mid = 0.5 * (breaks(1:end-1) + breaks(2:end));
span = sum(knots <= mid.', 2).';
connectivity = (span - p) + (0:p).';

supp = cell(1, ndof);
for e = 1:nel
    for a = 1:p + 1
        supp{connectivity(a, e)}(end + 1) = e;
    end
end

[xg, wg] = iga_gauss_legendre(nquad);
h = diff(breaks);
qn = breaks(1:end-1) + 0.5 * (xg + 1) * h;      % [nquad x nel]
qw = 0.5 * wg * h;

shp = zeros(nquad, p + 1, nel);
dshp = zeros(nquad, p + 1, nel);
for e = 1:nel
    [Ne, dNe] = iga_bspline_basis(knots, p, qn(:, e));
    shp(:, :, e) = Ne(:, connectivity(:, e));
    dshp(:, :, e) = dNe(:, connectivity(:, e));
end

s = struct('knots', knots, 'degree', p, 'ndof', ndof, 'nel', nel, ...
    'breaks', breaks, 'connectivity', connectivity, 'nquad', nquad, ...
    'qn', qn, 'qw', qw, 'shape_functions', shp, 'shape_function_gradients', dshp);
s.supp = supp;
end
