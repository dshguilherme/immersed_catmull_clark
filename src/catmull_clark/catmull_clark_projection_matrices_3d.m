function [P1, P2, P3] = catmull_clark_projection_matrices_3d(nelx, nely, nelz, degree)
% CATMULL_CLARK_PROJECTION_MATRICES_3D
% Generates 1D Catmull-Clark limit basis projection matrices for 3D regular grids.
% On regular hexahedral lattices, the Catmull-Clark / subdivision basis
% factors identically as tensor-products of 1D B-spline functions with
% C^{degree-1} inter-element continuity.
%
% Inputs:
%   nelx, nely, nelz - Number of elements along X, Y, Z
%   degree           - Polynomial degree (default: 2 for quadratic / 3 for cubic)
%
% Outputs:
%   P1, P2, P3       - 1D projection matrices evaluating basis at element centers

if nargin < 4 || isempty(degree), degree = 2; end

P1 = build_1d_cc_projection_3d(nelx, degree);
P2 = build_1d_cc_projection_3d(nely, degree);
P3 = build_1d_cc_projection_3d(nelz, degree);

end

function P = build_1d_cc_projection_3d(nel, p)
ncp = nel + p;
knots = [zeros(1, p+1), (1:(nel-1))/nel, ones(1, p+1)];
elem_centers = ((1:nel) - 0.5) / nel;

P = bspeval(p, eye(ncp), knots, elem_centers)'; % [nel x ncp]
end
