function [x, w] = iga_gauss_legendre(n)
% IGA_GAUSS_LEGENDRE  n-point Gauss-Legendre rule on [-1, 1] (Golub-Welsch).
%   [x, w] = iga_gauss_legendre(n) returns column vectors of nodes and weights.
%   Exact for polynomials of degree <= 2n-1.

if n == 1
    x = 0; w = 2;
    return;
end
k = (1:n-1).';
beta = k ./ sqrt(4 * k.^2 - 1);
[V, D] = eig(diag(beta, 1) + diag(beta, -1));
[x, idx] = sort(diag(D));
w = 2 * V(1, idx).'.^2;
end
