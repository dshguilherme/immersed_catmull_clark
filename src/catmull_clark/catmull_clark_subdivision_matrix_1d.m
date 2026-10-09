function [P_sub, P_3d] = catmull_clark_subdivision_matrix_1d()
% CATMULL_CLARK_SUBDIVISION_MATRIX_1D
% Returns the 1D and 3D dyadic subdivision projection matrices for
% uniform Catmull-Clark cubic B-spline basis functions.
%
% Outputs:
%   P_sub - 1D subdivision matrix [5 x 4] mapping 4 coarse control points
%           to 5 fine control points covering the two refined sub-intervals
%   P_3d  - 3D Kronecker tensor product subdivision matrix [125 x 64]
%           mapping 64 coarse control points to the 125 fine control points
%           of the 8 subdivided octant children

% 1D cubic B-spline dyadic subdivision mask: 1/8 * [1, 4, 6, 4, 1]
% Evaluated across the two half-intervals:
P_sub = 1/8 * [4, 4, 0, 0; ...
               1, 6, 1, 0; ...
               0, 4, 4, 0; ...
               0, 1, 6, 1; ...
               0, 0, 4, 4];

% 3D Kronecker product for regular hexahedral cells
P_3d = kron(P_sub, kron(P_sub, P_sub));

end
