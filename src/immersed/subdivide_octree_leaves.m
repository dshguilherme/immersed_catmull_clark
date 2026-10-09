function octree = subdivide_octree_leaves(octree, marked_leaf_indices)
% SUBDIVIDE_OCTREE_LEAVES
% Subdivides specified leaf elements of a 3D octree and applies 2:1 balancing.
%
% Inputs:
%   octree               - Octree struct (from octree_mesh_3d or balance_octree_3d)
%   marked_leaf_indices  - Vector of leaf indices (1 to octree.n_leaves) to subdivide
%
% Outputs:
%   octree               - Updated balanced octree struct

if isempty(marked_leaf_indices)
    return;
end

cells = octree.cells;
n_cells = numel(cells);
leaf_ids = octree.leaf_ids;

target_cell_ids = leaf_ids(marked_leaf_indices);

for idx = target_cell_ids(:)'
    if ~cells(idx).is_leaf
        continue;
    end
    
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

% Enforce 2:1 balancing
octree = balance_octree_3d(octree);

end
