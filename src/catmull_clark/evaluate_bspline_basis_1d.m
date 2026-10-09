function val = evaluate_bspline_basis_1d(degree, knots, u)
% EVALUATE_BSPLINE_BASIS_1D
% Clean-room in-house implementation of Cox-de Boor recursive algorithm
% for 1D B-spline basis function evaluation.
% Zero dependencies on external toolboxes or GeoPDEs / NURBS packages.
%
% Inputs:
%   degree - Polynomial degree p (e.g. 0, 1, 2, 3)
%   knots  - Non-decreasing knot vector [1 x (ncp + p + 1)]
%   u      - Parametric evaluation coordinates [nu x 1] or [1 x nu]
%
% Output:
%   val    - Matrix of size [nu x ncp] where val(k, i) = N_{i, p}(u_k)

u = u(:);
nu = numel(u);
nk = numel(knots);
p = degree;
ncp = nk - p - 1;

if ncp <= 0
    error('Knot vector length (%d) must exceed degree + 1 (%d)', nk, p + 1);
end

% Degree 0 basis functions
N = zeros(nu, nk - 1);
for i = 1:(nk - 1)
    if i == nk - 1
        % Include right endpoint for closed intervals
        N(:, i) = double(u >= knots(i) & u <= knots(i+1));
    else
        N(:, i) = double(u >= knots(i) & u < knots(i+1));
    end
end

% Cox-de Boor recursion up to target degree p
for d = 1:p
    N_next = zeros(nu, nk - d - 1);
    for i = 1:(nk - d - 1)
        denom1 = knots(i + d) - knots(i);
        denom2 = knots(i + d + 1) - knots(i + 1);
        
        term1 = zeros(nu, 1);
        term2 = zeros(nu, 1);
        
        if denom1 > 1e-15
            term1 = ((u - knots(i)) ./ denom1) .* N(:, i);
        end
        if denom2 > 1e-15
            term2 = ((knots(i + d + 1) - u) ./ denom2) .* N(:, i + 1);
        end
        
        N_next(:, i) = term1 + term2;
    end
    N = N_next;
end

val = N(:, 1:ncp);

end
