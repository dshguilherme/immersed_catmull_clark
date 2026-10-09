function [F_neumann, info] = assemble_neumann_bc_3d(mesh, brep, facet_indices, traction_spec, opts)
% ASSEMBLE_NEUMANN_BC_3D
% Evaluates consistent boundary work vector for distributed surface tractions
% or pressures on immersed CAD B-Rep surfaces embedded in an adaptive octree.
%
% Inputs:
%   mesh           - Structural octree mesh (from octree_structural_mesh)
%   brep           - CAD B-Rep surface mesh (.nodes, .elements)
%   facet_indices  - Vector of indices into brep.elements belonging to Gamma_N (empty = all)
%   traction_spec  - Tractions specification:
%                      - 3-element vector [tx, ty, tz] (constant surface traction)
%                      - Scalar P (normal pressure, t_N = -P * n_outward)
%                      - Function handle @(x, y, z, n) returning [tx, ty, tz]
%   opts           - (Optional) struct with quadrature settings
%
% Outputs:
%   F_neumann      - Master load vector of size [3*N_master x 1]
%   info           - Struct with integration diagnostics (total area, resultant force)

if nargin < 3 || isempty(facet_indices)
    facet_indices = (1:size(brep.elements, 1))';
end
if nargin < 5, opts = struct(); end

n_facets = numel(facet_indices);
if n_facets == 0
    F_neumann = zeros(3 * mesh.n_master, 1);
    info.total_area = 0;
    info.resultant_force = [0; 0; 0];
    return;
end

% Extract facet geometry
elems_N = brep.elements(facet_indices, :);
v1 = brep.nodes(elems_N(:, 1), :);
v2 = brep.nodes(elems_N(:, 2), :);
v3 = brep.nodes(elems_N(:, 3), :);

cross_n = cross(v2 - v1, v3 - v1, 2);
areas = 0.5 * sqrt(sum(cross_n.^2, 2));
unit_normals = cross_n ./ max(2 * areas, 1e-12);
centroids = (v1 + v2 + v3) / 3.0;

% Evaluate surface tractions at facet centroids
tractions = zeros(n_facets, 3);
if isnumeric(traction_spec) && numel(traction_spec) == 3
    % Constant traction vector
    tractions = repmat(traction_spec(:)', n_facets, 1);
elseif isnumeric(traction_spec) && isscalar(traction_spec)
    % Scalar normal pressure: t = -P * n
    P = traction_spec;
    tractions = -P * unit_normals;
elseif isa(traction_spec, 'function_handle')
    for i = 1:n_facets
        tractions(i, :) = traction_spec(centroids(i, 1), centroids(i, 2), centroids(i, 3), unit_normals(i, :));
    end
else
    error('traction_spec must be a 3-element vector, scalar pressure, or function handle.');
end

% Locate containing octree leaf cells
bounds = mesh.bounds;
n_leaves = mesh.n_elements;

xi_c  = [-1,  1,  1, -1, -1,  1,  1, -1];
eta_c = [-1, -1,  1,  1, -1, -1,  1,  1];
zt_c  = [-1, -1, -1, -1,  1,  1,  1,  1];

F_uncon = zeros(3 * mesh.n_nodes, 1);

for f = 1:n_facets
    pt = centroids(f, :);
    Af = areas(f);
    tf = tractions(f, :)';
    
    % Find containing leaf cell
    e_idx = find(pt(1) >= bounds(:,1)-1e-5 & pt(1) <= bounds(:,2)+1e-5 & ...
                 pt(2) >= bounds(:,3)-1e-5 & pt(2) <= bounds(:,4)+1e-5 & ...
                 pt(3) >= bounds(:,5)-1e-5 & pt(3) <= bounds(:,6)+1e-5, 1);
             
    if isempty(e_idx), continue; end
    
    b_el = bounds(e_idx, :);
    h_el = b_el([2 4 6]) - b_el([1 3 5]);
    c_el = 0.5 * (b_el([1 3 5]) + b_el([2 4 6]));
    
    xi  = max(-1, min(1, 2 * (pt(1) - c_el(1)) / h_el(1)));
    eta = max(-1, min(1, 2 * (pt(2) - c_el(2)) / h_el(2)));
    zt  = max(-1, min(1, 2 * (pt(3) - c_el(3)) / h_el(3)));
    
    N = 0.125 * (1 + xi_c * xi) .* (1 + eta_c * eta) .* (1 + zt_c * zt);
    
    en = mesh.elem_nodes(e_idx, :);
    for a = 1:8
        node_id = en(a);
        dof_start = (node_id - 1) * 3;
        F_uncon(dof_start + (1:3)) = F_uncon(dof_start + (1:3)) + (Af * N(a)) * tf;
    end
end

% Project to master system: F_master = T_3d' * F_uncon
F_neumann = mesh.T_3d' * F_uncon;

info.total_area = sum(areas);
info.resultant_force = sum(tractions .* areas, 1)';
info.n_facets = n_facets;

end
