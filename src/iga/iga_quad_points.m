function [X, W] = iga_quad_points(sp)
% IGA_QUAD_POINTS  Physical Gauss points and weights of every element.
%   [X, W] = iga_quad_points(sp)
%     X  {1 x dim} cell, X{d} [nqn x nel] physical coordinate d of each point
%     W  [nqn x nel] physical quadrature weights (Jacobian included)
%   Points within an element are lexicographic, direction 1 fastest.

dim = sp.dim;
sub = cell(1, dim);
[sub{:}] = ind2sub(sp.nel_dir, 1:sp.nel);
qloc = cell(1, dim);
[qloc{:}] = ind2sub(sp.nquad, (1:sp.nqn).');
X = cell(1, dim);
W = ones(sp.nqn, sp.nel);
for d = 1:dim
    u = sp.univ(d);
    qn = sp.bounds(d, 1) + sp.L(d) * u.qn;      % [nq_d x nel_d]
    qw = sp.L(d) * u.qw;
    X{d} = qn(sub2ind(size(qn), repmat(qloc{d}, 1, sp.nel), repmat(sub{d}, sp.nqn, 1)));
    W = W .* qw(sub2ind(size(qw), repmat(qloc{d}, 1, sp.nel), repmat(sub{d}, sp.nqn, 1)));
end
end
