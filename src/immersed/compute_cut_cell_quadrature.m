function [elem_weights, q_points] = compute_cut_cell_quadrature(brep, cell_bounds, subcell_res)
% COMPUTE_CUT_CELL_QUADRATURE Computes sub-cell quadrature weights for an immersed
% cut element using recursive Cartesian tessellation.
%
% Inputs:
%   brep         - B-Rep surface mesh (.nodes, .elements)
%   cell_bounds  - [xmin, xmax; ymin, ymax; zmin, zmax] for the background element
%   subcell_res  - [sx, sy, sz] sub-cell subdivision resolution (default: [4, 4, 4])
%
% Outputs:
%   elem_weights - Scalar volume fraction / effective integration weight inside the domain [0, 1]
%   q_points     - Sub-cell quadrature points located inside the domain

if nargin < 3 || isempty(subcell_res), subcell_res = [4, 4, 4]; end

sx = subcell_res(1); sy = subcell_res(2); sz = subcell_res(3);
xmin = cell_bounds(1,1); xmax = cell_bounds(1,2);
ymin = cell_bounds(2,1); ymax = cell_bounds(2,2);
zmin = cell_bounds(3,1); zmax = cell_bounds(3,2);

% 1D Gauss points and weights on reference interval [-1, 1] mapped to subcells
% Midpoints of sub-cells as 1st-order quadrature (or 2x2x2 Gauss sub-cell)
dx = (xmax - xmin) / sx;
dy = (ymax - ymin) / sy;
dz = (zmax - zmin) / sz;

sub_xc = xmin + ( (1:sx) - 0.5 ) * dx;
sub_yc = ymin + ( (1:sy) - 0.5 ) * dy;
sub_zc = zmin + ( (1:sz) - 0.5 ) * dz;

[Xq, Yq, Zq] = ndgrid(sub_xc, sub_yc, sub_zc);
candidates = [Xq(:), Yq(:), Zq(:)];

% Point-in-polyhedron parity test for each sub-cell point
inside_mask = in_polyhedron_points(brep.elements, brep.nodes, candidates);

% Total fraction of element volume inside B-Rep
n_total = sx * sy * sz;
elem_weights = sum(inside_mask) / n_total;
q_points = candidates(inside_mask, :);

end

function inside = in_polyhedron_points(faces, vertices, queries)
nq = size(queries, 1);
inside = false(nq, 1);

v1 = vertices(faces(:, 1), :);
v2 = vertices(faces(:, 2), :);
v3 = vertices(faces(:, 3), :);

minY = min(min(v1(:,2), v2(:,2)), v3(:,2));
maxY = max(max(v1(:,2), v2(:,2)), v3(:,2));
minZ = min(min(v1(:,3), v2(:,3)), v3(:,3));
maxZ = max(max(v1(:,3), v2(:,3)), v3(:,3));

for q = 1:nq
    qx = queries(q, 1); qy = queries(q, 2); qz = queries(q, 3);
    
    cand = (qy >= minY & qy <= maxY & qz >= minZ & qz <= maxZ);
    if ~any(cand), continue; end
    
    cand_f = find(cand);
    A = v1(cand_f, :); B = v2(cand_f, :); C = v3(cand_f, :);
    
    v0 = C(:, 2:3) - A(:, 2:3);
    v1_sub = B(:, 2:3) - A(:, 2:3);
    v2_sub = [qy, qz] - A(:, 2:3);
    
    dot00 = sum(v0.^2, 2);
    dot01 = sum(v0 .* v1_sub, 2);
    dot02 = sum(v0 .* v2_sub, 2);
    dot11 = sum(v1_sub.^2, 2);
    dot12 = sum(v1_sub .* v2_sub, 2);
    
    invDenom = 1 ./ (dot00 .* dot11 - dot01.^2 + 1e-15);
    u = (dot11 .* dot02 - dot01 .* dot12) .* invDenom;
    v = (dot00 .* dot12 - dot01 .* dot02) .* invDenom;
    
    hit = (u >= 0) & (v >= 0) & (u + v <= 1);
    if ~any(hit), continue; end
    
    hit_idx = cand_f(hit);
    Ah = v1(hit_idx, :); Bh = v2(hit_idx, :); Ch = v3(hit_idx, :);
    uh = u(hit); vh = v(hit);
    
    x_hit = Ah(:,1) + vh .* (Bh(:,1) - Ah(:,1)) + uh .* (Ch(:,1) - Ah(:,1));
    intersections = sum(x_hit > qx);
    
    inside(q) = (mod(intersections, 2) == 1);
end
end
