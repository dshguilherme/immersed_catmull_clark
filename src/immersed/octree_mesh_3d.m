function octree = octree_mesh_3d(grid_bounds, grid_res, max_level, refine_criteria_fn)
% OCTREE_MESH_3D
% Constructs an adaptive 3D octree mesh with 2:1 balancing for immersed IGA
% and Catmull-Clark local subdivision refinement.
%
% Inputs:
%   grid_bounds        - [xmin, xmax; ymin, ymax; zmin, zmax]
%   grid_res           - [nx, ny, nz] root level resolution
%   max_level          - Maximum octree depth (0 = root only, 1 = 1 subdivision, etc.)
%   refine_criteria_fn - Function handle @(cell_bounds) returning true if cell should be refined
%
% Outputs:
%   octree             - Struct containing:
%                        .cells      - Struct array of all tree nodes
%                        .leaf_ids   - Indices of active leaf cells
%                        .n_leaves   - Total leaf count
%                        .bounds     - Leaf bounding boxes [n_leaves x 6]
%                        .levels     - Leaf depth levels [n_leaves x 1]

if nargin < 3, max_level = 1; end
if nargin < 4 || isempty(refine_criteria_fn)
    refine_criteria_fn = @(b) false;
end

nx = grid_res(1); ny = grid_res(2); nz = grid_res(3);
n_root = nx * ny * nz;

xmin = grid_bounds(1,1); xmax = grid_bounds(1,2);
ymin = grid_bounds(2,1); ymax = grid_bounds(2,2);
zmin = grid_bounds(3,1); zmax = grid_bounds(3,2);

hx = (xmax - xmin) / nx;
hy = (ymax - ymin) / ny;
hz = (zmax - zmin) / nz;

% Allocate cells
% Each cell: .bounds [xmin xmax ymin ymax zmin zmax], .level, .is_leaf, .children (1x8)
cells = struct('bounds', cell(n_root * 10, 1), ...
               'level', cell(n_root * 10, 1), ...
               'is_leaf', cell(n_root * 10, 1), ...
               'children', cell(n_root * 10, 1), ...
               'parent', cell(n_root * 10, 1), ...
               'ijk', cell(n_root * 10, 1));

n_cells = 0;

% Initialize root level cells
for k = 1:nz
    for j = 1:ny
        for i = 1:nx
            n_cells = n_cells + 1;
            cb = [xmin + (i-1)*hx, xmin + i*hx, ...
                  ymin + (j-1)*hy, ymin + j*hy, ...
                  zmin + (k-1)*hz, zmin + k*hz];
            cells(n_cells).bounds = cb;
            cells(n_cells).level = 0;
            cells(n_cells).is_leaf = true;
            cells(n_cells).children = [];
            cells(n_cells).parent = 0;
            cells(n_cells).ijk = [i, j, k];
        end
    end
end

% Recursive refinement loop
for lvl = 0:(max_level - 1)
    current_leaves = find([cells(1:n_cells).is_leaf] & [cells(1:n_cells).level] == lvl);
    for idx = current_leaves
        cb = cells(idx).bounds;
        if refine_criteria_fn([cb(1), cb(2); cb(3), cb(4); cb(5), cb(6)])
            % Subdivide into 8 octants
            xmid = 0.5 * (cb(1) + cb(2));
            ymid = 0.5 * (cb(3) + cb(4));
            zmid = 0.5 * (cb(5) + cb(6));
            
            x_spans = [cb(1), xmid; xmid, cb(2)];
            y_spans = [cb(3), ymid; ymid, cb(4)];
            z_spans = [cb(5), zmid; zmid, cb(6)];
            
            child_ids = zeros(1, 8);
            c_idx = 0;
            for zk = 1:2
                for yk = 1:2
                    for xk = 1:2
                        c_idx = c_idx + 1;
                        n_cells = n_cells + 1;
                        child_ids(c_idx) = n_cells;
                        
                        cells(n_cells).bounds = [x_spans(xk,1), x_spans(xk,2), ...
                                                 y_spans(yk,1), y_spans(yk,2), ...
                                                 z_spans(zk,1), z_spans(zk,2)];
                        cells(n_cells).level = lvl + 1;
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

% Extract active leaf cells
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
octree.grid_bounds = grid_bounds;
octree.grid_res = grid_res;

% Enforce 2:1 balancing across all leaf elements
octree = balance_octree_3d(octree);

fprintf('Constructed 3D Octree Mesh: %d total cells, %d active leaf elements (Max Level: %d)\n', ...
    numel(octree.cells), octree.n_leaves, max(octree.levels));

end
