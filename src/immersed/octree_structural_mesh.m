function mesh = octree_structural_mesh(octree_in, brep, opts)
% OCTREE_STRUCTURAL_MESH
% Builds an analysis-ready 3D structural mechanics mesh on an adaptive octree
% with Multi-Point Constraint (MPC) hanging-node elimination and immersed
% cut-cell stiffness evaluation.
%
% Inputs:
%   octree_in - Octree struct (from octree_mesh_3d / balance_octree_3d)
%   brep      - (Optional) B-Rep surface mesh (.nodes, .elements)
%   opts      - (Optional) Struct with configuration:
%                 .E               - Young's modulus (default: 1.0e5)
%                 .nu              - Poisson's ratio (default: 0.3)
%                 .fictitious_weight - Weight for exterior cells (default: 1e-4)
%                 .subcell_res     - [sx, sy, sz] cut-cell quadrature (default: [3, 3, 3])
%
% Outputs:
%   mesh      - Struct containing:
%                 .octree          - Balanced octree struct
%                 .nodes           - [N_nodes x 3] unique node coordinates
%                 .elem_nodes      - [N_elem x 8] connectivity matrix
%                 .n_nodes         - Total unique nodes
%                 .n_elements      - Number of leaf elements
%                 .is_hanging      - [N_nodes x 1] logical hanging flags
%                 .master_node_ids - [N_master x 1] master node indices
%                 .n_master        - Number of master nodes
%                 .T               - [N_nodes x N_master] scalar constraint matrix
%                 .T_3d            - [3*N_nodes x 3*N_master] 3D constraint matrix
%                 .K_uncon         - [3*N_nodes x 3*N_nodes] unconstrained stiffness
%                 .K_master        - [3*N_master x 3*N_master] constrained stiffness
%                 .weights         - [N_elem x 1] element volume weights
%                 .C_tensor        - [6 x 6] constitutive tensor
%                 .h_elem          - [N_elem x 3] element sizes

if nargin < 2, brep = []; end
if nargin < 3, opts = struct(); end

if ~isfield(opts, 'E'), opts.E = 1.0e5; end
if ~isfield(opts, 'nu'), opts.nu = 0.3; end
if ~isfield(opts, 'fictitious_weight'), opts.fictitious_weight = 1e-4; end
if ~isfield(opts, 'subcell_res'), opts.subcell_res = [3, 3, 3]; end

octree = balance_octree_3d(octree_in);

n_leaves = octree.n_leaves;
bounds = octree.bounds;
levels = octree.levels;

% 1. Extract element corners and unique nodes
raw_verts = zeros(n_leaves * 8, 3);
for e = 1:n_leaves
    b = bounds(e, :);
    v = [b(1) b(3) b(5);  % 1: -x -y -z
         b(2) b(3) b(5);  % 2: +x -y -z
         b(2) b(4) b(5);  % 3: +x +y -z
         b(1) b(4) b(5);  % 4: -x +y -z
         b(1) b(3) b(6);  % 5: -x -y +z
         b(2) b(3) b(6);  % 6: +x -y +z
         b(2) b(4) b(6);  % 7: +x +y +z
         b(1) b(4) b(6)]; % 8: -x +y +z
    raw_verts((e-1)*8 + (1:8), :) = v;
end

h_mins = min(bounds(:, [2 4 6]) - bounds(:, [1 3 5]), [], 1);
tol = 1e-5 * min(h_mins);

[unique_nodes, ~, elem_nodes_flat] = unique(round(raw_verts / tol) * tol, 'rows', 'stable');
elem_nodes = reshape(elem_nodes_flat, 8, n_leaves)';
N_nodes = size(unique_nodes, 1);

% 2. Identify hanging nodes and Multi-Point Constraints
hex_edges = [1 2; 2 3; 3 4; 4 1; 5 6; 6 7; 7 8; 8 5; 1 5; 2 6; 3 7; 4 8];
hex_faces = [1 4 3 2; 5 6 7 8; 1 2 6 5; 4 8 7 3; 1 5 8 4; 2 3 7 6];

hash_nodes = round(unique_nodes / tol);
hanging = false(N_nodes, 1);
C_mat = speye(N_nodes);

% Vectorized search for edge midpoints
all_edge_n1 = elem_nodes(:, hex_edges(:,1));
all_edge_n2 = elem_nodes(:, hex_edges(:,2));
p_mids = 0.5 * (unique_nodes(all_edge_n1(:), :) + unique_nodes(all_edge_n2(:), :));
keys_mid = round(p_mids / tol);
[tf_mid, loc_mid] = ismember(keys_mid, hash_nodes, 'rows');

mid_match = find(tf_mid);
n1_arr = all_edge_n1(:);
n2_arr = all_edge_n2(:);

for idx = mid_match'
    matched_node = loc_mid(idx);
    na = n1_arr(idx);
    nb = n2_arr(idx);
    if matched_node ~= na && matched_node ~= nb
        hanging(matched_node) = true;
        C_mat(matched_node, :) = 0;
        C_mat(matched_node, na) = 0.5;
        C_mat(matched_node, nb) = 0.5;
    end
end

% Vectorized search for face centers
all_face_fn = cell(6, 1);
for fc = 1:6
    fn_fc = elem_nodes(:, hex_faces(fc, :));
    p_cen = 0.25 * (unique_nodes(fn_fc(:,1), :) + unique_nodes(fn_fc(:,2), :) + ...
                    unique_nodes(fn_fc(:,3), :) + unique_nodes(fn_fc(:,4), :));
    keys_cen = round(p_cen / tol);
    [tf_cen, loc_cen] = ismember(keys_cen, hash_nodes, 'rows');
    cen_match = find(tf_cen);
    for idx = cen_match'
        matched_node = loc_cen(idx);
        f_nodes = fn_fc(idx, :);
        if ~ismember(matched_node, f_nodes)
            hanging(matched_node) = true;
            C_mat(matched_node, :) = 0;
            C_mat(matched_node, f_nodes) = 0.25;
        end
    end
end

% Propagate constraints for multi-level trees
for iter = 1:5
    C_mat = C_mat * C_mat;
end

master_ids = find(~hanging);
N_master = numel(master_ids);
T = C_mat(:, master_ids);

% Expand T to 3D interleaved degrees of freedom: [u_x, u_y, u_z]
[ti, tj, tv] = find(T);
ti_3d = [(ti-1)*3 + 1, (ti-1)*3 + 2, (ti-1)*3 + 3]';
tj_3d = [(tj-1)*3 + 1, (tj-1)*3 + 2, (tj-1)*3 + 3]';
tv_3d = [tv, tv, tv]';

T_3d = sparse(ti_3d(:), tj_3d(:), tv_3d(:), 3*N_nodes, 3*N_master);

% 3. Immersed volume weights
weights = ones(n_leaves, 1);
if ~isempty(brep)
    fprintf('Evaluating cut-cell weights for %d octree elements...\n', n_leaves);
    t_w = tic;
    % Fast element center test
    el_cents = [0.5*(bounds(:,1)+bounds(:,2)), ...
                0.5*(bounds(:,3)+bounds(:,4)), ...
                0.5*(bounds(:,5)+bounds(:,6))];
            
    % Estimate weights using subcell integration
    for e = 1:n_leaves
        cb = [bounds(e,1), bounds(e,2); ...
              bounds(e,3), bounds(e,4); ...
              bounds(e,5), bounds(e,6)];
        w_cut = compute_cut_cell_quadrature(brep, cb, opts.subcell_res);
        weights(e) = max(w_cut, opts.fictitious_weight);
    end
    fprintf('Element weights evaluated in %.2f s.\n', toc(t_w));
end

% 4. Constitutive matrix
E = opts.E; nu = opts.nu;
lam = E * nu / ((1 + nu) * (1 - 2*nu));
mu = E / (2 * (1 + nu));
C_tensor = [lam+2*mu, lam, lam, 0, 0, 0;
            lam, lam+2*mu, lam, 0, 0, 0;
            lam, lam, lam+2*mu, 0, 0, 0;
            0, 0, 0, mu, 0, 0;
            0, 0, 0, 0, mu, 0;
            0, 0, 0, 0, 0, mu];

% Precompute standard element stiffness matrices per unique level
unique_levels = unique(levels);
ke_by_level = cell(max(unique_levels)+1, 1);

gp = [-1, 1] / sqrt(3);
gw = [1, 1];
xi_c  = [-1,  1,  1, -1, -1,  1,  1, -1];
eta_c = [-1, -1,  1,  1, -1, -1,  1,  1];
zt_c  = [-1, -1, -1, -1,  1,  1,  1,  1];

root_h = [bounds(1,2)-bounds(1,1), bounds(1,4)-bounds(1,3), bounds(1,6)-bounds(1,5)] * (2^levels(1));

for ul = unique_levels(:)'
    h_lvl = root_h / (2^ul);
    ke_lvl = zeros(24, 24);
    detJ = prod(h_lvl) / 8;
    
    for ix = 1:2
        for iy = 1:2
            for iz = 1:2
                xi = gp(ix); eta = gp(iy); zt = gp(iz);
                w = gw(ix) * gw(iy) * gw(iz);
                
                dNdxi  = (2/h_lvl(1)) * 0.125 * xi_c  .* (1 + eta_c * eta) .* (1 + zt_c * zt);
                dNdeta = (2/h_lvl(2)) * 0.125 * eta_c .* (1 + xi_c * xi)   .* (1 + zt_c * zt);
                dNdzt  = (2/h_lvl(3)) * 0.125 * zt_c  .* (1 + xi_c * xi)   .* (1 + eta_c * eta);
                
                B = zeros(6, 24);
                for a = 1:8
                    col = (a-1)*3 + (1:3);
                    B(:, col) = [dNdxi(a), 0, 0;
                                 0, dNdeta(a), 0;
                                 0, 0, dNdzt(a);
                                 0, dNdzt(a), dNdeta(a);
                                 dNdzt(a), 0, dNdxi(a);
                                 dNdeta(a), dNdxi(a), 0];
                end
                ke_lvl = ke_lvl + (B' * C_tensor * B) * (detJ * w);
            end
        end
    end
    ke_by_level{ul+1} = 0.5 * (ke_lvl + ke_lvl');
end

% 5. Global assembly
elem_dofs = zeros(n_leaves, 24);
for a = 1:8
    elem_dofs(:, (a-1)*3 + (1:3)) = [(elem_nodes(:, a)-1)*3 + 1, ...
                                     (elem_nodes(:, a)-1)*3 + 2, ...
                                     (elem_nodes(:, a)-1)*3 + 3];
end

rows_ke = repmat(elem_dofs, 1, 24)';
cols_ke = repelem(elem_dofs, 1, 24)';
vals_ke = zeros(24*24, n_leaves);

for e = 1:n_leaves
    lvl = levels(e);
    ke_base = ke_by_level{lvl+1};
    vals_ke(:, e) = weights(e) * ke_base(:);
end

K_uncon = sparse(rows_ke(:), cols_ke(:), vals_ke(:), 3*N_nodes, 3*N_nodes);
K_uncon = 0.5 * (K_uncon + K_uncon');

% 6. Project to master system
K_master = T_3d' * K_uncon * T_3d;
K_master = 0.5 * (K_master + K_master');

% Populate output mesh struct
mesh.octree = octree;
mesh.nodes = unique_nodes;
mesh.elem_nodes = elem_nodes;
mesh.n_nodes = N_nodes;
mesh.n_elements = n_leaves;
mesh.is_hanging = hanging;
mesh.master_node_ids = master_ids;
mesh.n_master = N_master;
mesh.T = T;
mesh.T_3d = T_3d;
mesh.K_uncon = K_uncon;
mesh.K_master = K_master;
mesh.weights = weights;
mesh.C_tensor = C_tensor;
mesh.bounds = bounds;
mesh.levels = levels;
mesh.h_elem = bounds(:, [2 4 6]) - bounds(:, [1 3 5]);

fprintf('Structural Octree Mesh built: %d elements, %d nodes (%d master, %d hanging), %d DOFs\n', ...
    n_leaves, N_nodes, N_master, sum(hanging), 3*N_master);

end
