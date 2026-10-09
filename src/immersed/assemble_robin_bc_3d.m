function [K_robin, F_robin, info] = assemble_robin_bc_3d(mesh, brep, facet_indices, k_spring, traction_spec, opts)
% ASSEMBLE_ROBIN_BC_3D
% Evaluates consistent stiffness matrix and load vector for elastic foundation /
% impedance / Robin boundary conditions:
%   sigma * n + k_s * u = t_R   on Gamma_R
%
% Weak form contribution:
%   a_R(u, v) = \int_{\Gamma_R} v' * K_s * u d\Gamma
%   l_R(v)    = \int_{\Gamma_R} v' * t_R d\Gamma
%
% Inputs:
%   mesh          - Structural octree mesh (from octree_structural_mesh)
%   brep          - CAD B-Rep surface mesh (.nodes, .elements)
%   facet_indices - Vector of indices into brep.elements belonging to Gamma_R (empty = all)
%   k_spring      - Foundation stiffness:
%                     - Scalar k_s: isotropic foundation (K_s = k_s * I_3)
%                     - 3-element vector [kx, ky, kz]: anisotropic diagonal foundation
%                     - Struct with fields .kn, .kt: normal and tangential spring rates
%                     - 3x3 matrix: constant stiffness tensor
%   traction_spec - (Optional) Preload traction on Gamma_R:
%                     - 3-element vector [tx, ty, tz] (default: [0, 0, 0])
%                     - Scalar pressure P: t_R = -P * n
%                     - Function handle @(x,y,z,n)
%   opts          - (Optional) settings struct
%
% Outputs:
%   K_robin       - Master stiffness matrix [3*N_master x 3*N_master] (sparse symmetric)
%   F_robin       - Master load vector [3*N_master x 1]
%   info          - Diagnostics struct

if nargin < 3 || isempty(facet_indices)
    facet_indices = (1:size(brep.elements, 1))';
end
if nargin < 4 || isempty(k_spring)
    k_spring = 0.0;
end
if nargin < 5 || isempty(traction_spec)
    traction_spec = [0, 0, 0];
end
if nargin < 6, opts = struct(); end

n_master = mesh.n_master;
n_nodes  = mesh.n_nodes;
n_facets = numel(facet_indices);

if n_facets == 0
    K_robin = sparse(3 * n_master, 3 * n_master);
    F_robin = zeros(3 * n_master, 1);
    info.total_area = 0;
    info.n_facets = 0;
    return;
end

% Extract facet geometry
elems_R = brep.elements(facet_indices, :);
v1 = brep.nodes(elems_R(:, 1), :);
v2 = brep.nodes(elems_R(:, 2), :);
v3 = brep.nodes(elems_R(:, 3), :);

cross_n = cross(v2 - v1, v3 - v1, 2);
areas = 0.5 * sqrt(sum(cross_n.^2, 2));
unit_normals = cross_n ./ max(2 * areas, 1e-12);
centroids = (v1 + v2 + v3) / 3.0;

% Evaluate tractions
tractions = zeros(n_facets, 3);
if isnumeric(traction_spec) && numel(traction_spec) == 3
    tractions = repmat(traction_spec(:)', n_facets, 1);
elseif isnumeric(traction_spec) && isscalar(traction_spec)
    tractions = -traction_spec * unit_normals;
elseif isa(traction_spec, 'function_handle')
    for i = 1:n_facets
        tractions(i, :) = traction_spec(centroids(i, 1), centroids(i, 2), centroids(i, 3), unit_normals(i, :));
    end
end

% Standard trilinear corner signs
xi_c  = [-1,  1,  1, -1, -1,  1,  1, -1];
eta_c = [-1, -1,  1,  1, -1, -1,  1,  1];
zt_c  = [-1, -1, -1, -1,  1,  1,  1,  1];

bounds = mesh.bounds;

% Preallocate sparse triplet storage for unconstrained K
% 8 nodes x 3 dofs = 24 local dofs per facet => 24 x 24 = 576 entries
entries_per_facet = 24 * 24;
max_entries = n_facets * entries_per_facet;
i_list = zeros(max_entries, 1);
j_list = zeros(max_entries, 1);
s_list = zeros(max_entries, 1);
entry_ptr = 0;

F_uncon = zeros(3 * n_nodes, 1);

for f = 1:n_facets
    pt = centroids(f, :);
    Af = areas(f);
    nf = unit_normals(f, :)';
    tf = tractions(f, :)';
    
    % Determine local 3x3 spring matrix Ks
    if isnumeric(k_spring) && isscalar(k_spring)
        Ks = k_spring * eye(3);
    elseif isnumeric(k_spring) && numel(k_spring) == 3
        Ks = diag(k_spring(:));
    elseif isnumeric(k_spring) && all(size(k_spring) == [3, 3])
        Ks = k_spring;
    elseif isstruct(k_spring) && isfield(k_spring, 'kn') && isfield(k_spring, 'kt')
        % Normal and tangential decomposed springs
        kn = k_spring.kn;
        kt = k_spring.kt;
        P_n = nf * nf';
        P_t = eye(3) - P_n;
        Ks = kn * P_n + kt * P_t;
    else
        Ks = zeros(3, 3);
    end
    
    % Locate containing octree leaf element
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
    
    % Construct 24 local DOFs
    loc_dofs = zeros(24, 1);
    for a = 1:8
        loc_dofs((a-1)*3 + (1:3)) = (en(a) - 1) * 3 + (1:3);
    end
    
    % Local 24x24 block: Af * (N * N') kron Ks
    N_outer = N(:) * N(:)';  % 8x8
    Ke_block = Af * kron(N_outer, Ks); % 24x24
    
    % Symmetrize Ke_block to avoid floating-point drift
    Ke_block = 0.5 * (Ke_block + Ke_block');
    
    % Accumulate into triplets
    idx_start = entry_ptr + 1;
    idx_end = entry_ptr + entries_per_facet;
    
    [col_grid, row_grid] = meshgrid(loc_dofs, loc_dofs);
    i_list(idx_start:idx_end) = row_grid(:);
    j_list(idx_start:idx_end) = col_grid(:);
    s_list(idx_start:idx_end) = Ke_block(:);
    entry_ptr = idx_end;
    
    % Load vector contribution
    if norm(tf) > 0
        for a = 1:8
            node_id = en(a);
            dof_start = (node_id - 1) * 3;
            F_uncon(dof_start + (1:3)) = F_uncon(dof_start + (1:3)) + (Af * N(a)) * tf;
        end
    end
end

% Assemble unconstrained sparse matrix
if entry_ptr > 0
    K_uncon = sparse(i_list(1:entry_ptr), j_list(1:entry_ptr), s_list(1:entry_ptr), ...
                     3 * n_nodes, 3 * n_nodes);
else
    K_uncon = sparse(3 * n_nodes, 3 * n_nodes);
end

% Project to master system: K_robin = T_3d' * K_uncon * T_3d, F_robin = T_3d' * F_uncon
K_robin = mesh.T_3d' * (K_uncon * mesh.T_3d);
F_robin = mesh.T_3d' * F_uncon;

% Ensure exact symmetry
K_robin = 0.5 * (K_robin + K_robin');

info.total_area = sum(areas);
info.n_facets = n_facets;
info.trace_K = trace(K_robin);
info.resultant_force = sum(tractions .* areas, 1)';

end
