function status = classify_background_cells(mesh_brep, grid_bounds, grid_res)
% CLASSIFY_BACKGROUND_CELLS Classifies Cartesian/Catmull-Clark background cells
% against a 3D closed B-Rep surface mesh (e.g. NIST STEP model).
%
% Inputs:
%   mesh_brep   - Struct with .nodes (V x 3) and .elements (F x 3)
%   grid_bounds - [xmin, xmax; ymin, ymax; zmin, zmax]
%   grid_res    - [nx, ny, nz] number of background elements in X, Y, Z
%
% Outputs:
%   status - [nx, ny, nz] classification matrix:
%            1 : Inside cell (uncut interior)
%            0 : Outside cell (inactive exterior)
%           -1 : Cut cell (intersected by B-Rep boundary)

nx = grid_res(1); ny = grid_res(2); nz = grid_res(3);
xmin = grid_bounds(1,1); xmax = grid_bounds(1,2);
ymin = grid_bounds(2,1); ymax = grid_bounds(2,2);
zmin = grid_bounds(3,1); zmax = grid_bounds(3,2);

hx = (xmax - xmin) / nx;
hy = (ymax - ymin) / ny;
hz = (zmax - zmin) / nz;

% Coordinates of background cell vertices
[Xv, Yv, Zv] = ndgrid(linspace(xmin, xmax, nx+1), ...
                      linspace(ymin, ymax, ny+1), ...
                      linspace(zmin, zmax, nz+1));
pts = [Xv(:), Yv(:), Zv(:)];

% Evaluate inside/outside status of background vertices via ray-casting
inside_pts = inpolyhedron_fast(mesh_brep.elements, mesh_brep.nodes, pts);
inside_v = reshape(inside_pts, [nx+1, ny+1, nz+1]);

status = zeros(nx, ny, nz);

% An element is:
% - Inside (1) if all 8 corner vertices are inside
% - Outside (0) if all 8 corner vertices are outside
% - Cut (-1) if it contains both inside and outside vertices
for i = 1:nx
    for j = 1:ny
        for k = 1:nz
            corners = inside_v(i:i+1, j:j+1, k:k+1);
            s = sum(corners(:));
            if s == 8
                status(i, j, k) = 1;  % Completely inside
            elseif s == 0
                status(i, j, k) = 0;  % Completely outside
            else
                status(i, j, k) = -1; % Cut element
            end
        end
    end
end

fprintf('Classified %d background cells:\n  Inside: %d | Outside: %d | Cut: %d\n', ...
    numel(status), sum(status(:) == 1), sum(status(:) == 0), sum(status(:) == -1));

end

function inside = inpolyhedron_fast(faces, vertices, queries)
% Parity count ray-casting along +X direction
% Intersects ray (y_q, z_q) with triangular facets
nq = size(queries, 1);
inside = false(nq, 1);

v1 = vertices(faces(:, 1), :);
v2 = vertices(faces(:, 2), :);
v3 = vertices(faces(:, 3), :);

% Bounding box for facets in Y and Z
minY = min(min(v1(:,2), v2(:,2)), v3(:,2));
maxY = max(max(v1(:,2), v2(:,2)), v3(:,2));
minZ = min(min(v1(:,3), v2(:,3)), v3(:,3));
maxZ = max(max(v1(:,3), v2(:,3)), v3(:,3));

for q = 1:nq
    qx = queries(q, 1); qy = queries(q, 2); qz = queries(q, 3);
    
    % Candidate triangles matching Y and Z bounds
    cand = (qy >= minY & qy <= maxY & qz >= minZ & qz <= maxZ);
    if ~any(cand), continue; end
    
    cand_f = find(cand);
    A = v1(cand_f, :); B = v2(cand_f, :); C = v3(cand_f, :);
    
    % Barycentric 2D intersection test on Y-Z projection
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
    
    % Check if intersection X is greater than qx
    hit_idx = cand_f(hit);
    Ah = v1(hit_idx, :); Bh = v2(hit_idx, :); Ch = v3(hit_idx, :);
    uh = u(hit); vh = v(hit);
    
    % Compute x on triangle plane
    x_hit = Ah(:,1) + vh .* (Bh(:,1) - Ah(:,1)) + uh .* (Ch(:,1) - Ah(:,1));
    intersections = sum(x_hit > qx);
    
    inside(q) = (mod(intersections, 2) == 1);
end

end
