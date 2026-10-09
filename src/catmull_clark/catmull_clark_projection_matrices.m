function [P1, P2] = catmull_clark_projection_matrices(nelx, nely)
% CATMULL_CLARK_PROJECTION_MATRICES
% Generates the 1D Catmull-Clark cubic B-spline projection matrices
% for a regular grid of nelx x nely elements.
%
% On regular quadrilateral meshes, Catmull-Clark subdivision basis functions
% coincide with uniform cubic B-splines with C^2 inter-element continuity.
%
% Inputs:
%   nelx, nely - Number of elements in x and y directions
%
% Outputs:
%   P1 - [nelx x (nelx + 3)] projection matrix along X (evaluating CC basis at element centers)
%   P2 - [nely x (nely + 3)] projection matrix along Y (evaluating CC basis at element centers)

P1 = build_1d_cc_projection(nelx);
P2 = build_1d_cc_projection(nely);

end

function P = build_1d_cc_projection(nel)
% Standard open knot vector for cubic B-splines (Catmull-Clark limit on regular grid)
p = 3;
ncp = nel + p; % number of control points
knots = [zeros(1, p+1), (1:(nel-1))/nel, ones(1, p+1)];
elem_centers = ( (1:nel) - 0.5 ) / nel;

P = evaluate_bspline_basis_1d(p, knots, elem_centers); % [nel x ncp]
end
