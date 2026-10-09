function octree = balance_octree_3d(octree, tol)
% BALANCE_OCTREE_3D
% Enforces 2:1 balancing on a 3D octree mesh: no two adjacent leaf cells
% can differ by more than 1 subdivision level.
%
% Inputs:
%   octree - Octree struct (from octree_mesh_3d)
%   tol    - Coordinate tolerance for contact detection (default: 1e-6)
%
% Outputs:
%   octree - Balanced octree struct with updated cells, leaf_ids, bounds, levels

if nargin < 2 || isempty(tol), tol = 1e-6; end

cells = octree.cells;
n_cells = numel(cells);

balanced = false;
max_passes = 20;
pass = 0;

while ~balanced && pass < max_passes
    pass = pass + 1;
    balanced = true;
    
    leaf_ids = find([cells(1:n_cells).is_leaf]);
    n_leaves = numel(leaf_ids);
    
    bounds = zeros(n_leaves, 6);
    levels = zeros(n_leaves, 1);
    for l = 1:n_leaves
        bounds(l, :) = cells(leaf_ids(l)).bounds;
        levels(l) = cells(leaf_ids(l)).level;
    end
    
    to_subdivide = false(n_leaves, 1);
    
    % Vectorized bounding-box overlap test across all leaf pairs
    for i = 1:n_leaves
        lvl_i = levels(i);
        bi = bounds(i, :);
        
        % Only test against cells with level difference > 1
        cand = find(levels > (lvl_i + 1));
        if isempty(cand), continue; end
        
        bc = bounds(cand, :);
        touch = (bi(1) <= bc(:,2) + tol & bi(2) >= bc(:,1) - tol & ...
                 bi(3) <= bc(:,4) + tol & bi(4) >= bc(:,3) - tol & ...
                 bi(5) <= bc(:,6) + tol & bi(6) >= bc(:,5) - tol);
             
        if any(touch)
            to_subdivide(i) = true;
            balanced = false;
        end
    end
    
    if ~balanced
        refine_list = leaf_ids(to_subdivide);
        for idx = refine_list(:)'
            cb = cells(idx).bounds;
            xmid = 0.5 * (cb(1) + cb(2));
            ymid = 0.5 * (cb(3) + cb(4));
            zmid = 0.5 * (cb(5) + cb(6));
            
            x_spans = [cb(1), xmid; xmid, cb(2)];
            y_spans = [cb(3), ymid; ymid, cb(4)];
            z_spans = [cb(5), zmid; zmid, cb(6)];
            
            child_ids = zeros(1, 8);
            c_idx = 0;
            clvl = cells(idx).level;
            
            for zk = 1:2
                for yk = 1:2
                    for xk = 1:2
                        c_idx = c_idx + 1;
                        n_cells = n_cells + 1;
                        child_ids(c_idx) = n_cells;
                        
                        cells(n_cells).bounds = [x_spans(xk,1), x_spans(xk,2), ...
                                                 y_spans(yk,1), y_spans(yk,2), ...
                                                 z_spans(zk,1), z_spans(zk,2)];
                        cells(n_cells).level = clvl + 1;
                        cells(n_cells).is_leaf = true;
                        cells(n_cells).children = [];
                        cells(n_cells).parent = idx;
                        cells(n_cells).ijk = [xk, yk, zk];
                    end
                end
            end
            
            cells(idx).is_leaf = false;
            cells(idx).children = child_ids;
        end
    end
end

% Extract updated active leaves
leaf_ids = find([cells(1:n_cells).is_leaf]);
n_leaves = numel(leaf_ids);
leaf_bounds = zeros(n_leaves, 6);
leaf_levels = zeros(n_leaves, 1);

for l = 1:n_leaves
    cid = leaf_ids(l);
    leaf_bounds(l, :) = cells(cid).bounds;
    leaf_levels(l) = cells(cid).level;
end

octree.cells = cells(1:n_cells);
octree.leaf_ids = leaf_ids;
octree.n_leaves = n_leaves;
octree.bounds = leaf_bounds;
octree.levels = leaf_levels;

end
