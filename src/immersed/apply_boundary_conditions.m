function [K_bc, F_bc, bc_state] = apply_boundary_conditions(mesh, brep, bc_list, opts)
% APPLY_BOUNDARY_CONDITIONS
% Unified boundary condition manager and dispatcher for immersed octree IGA.
% Supports Dirichlet (strong elimination or weak penalty/Nitsche),
% Neumann (distributed surface tractions and normal pressure), and
% Robin (elastic foundation / impedance springs).
%
% Inputs:
%   mesh     - Structural octree mesh (from octree_structural_mesh)
%   brep     - CAD B-Rep surface mesh (.nodes, .elements)
%   bc_list  - Cell array or array of structs defining boundary conditions:
%              Each entry:
%                .type       - 'dirichlet' (or 'fixed'), 'neumann' (or 'traction'), 'robin'
%                .filter     - Function handle @(x,y,z), facet index vector, or bbox [xmin xmax; ymin ymax; zmin zmax]
%                .value      - Traction/displacement vector [val_x, val_y, val_z], scalar pressure, or handle
%                .components - [1 1 1] (default) or logical flag for [x, y, z] components
%                .method     - For dirichlet: 'strong' (default) or 'penalty'
%                .penalty    - Penalty stiffness parameter for weak Dirichlet (default: 1e8)
%                .k_spring   - For robin: spring stiffness (scalar, 3-vector, or struct with .kn, .kt)
%   opts     - (Optional) settings struct
%
% Outputs:
%   K_bc     - Master stiffness matrix with boundary conditions applied
%   F_bc     - Master load vector with boundary conditions applied
%   bc_state - Struct containing:
%                .fixed_dofs       - Vector of strongly constrained master DOFs
%                .prescribed_vals  - Values for strongly constrained master DOFs
%                .free_dofs        - Vector of free master DOFs
%                .K_unreduced      - Matrix before strong DOF elimination
%                .F_unreduced      - Vector before strong DOF elimination
%                .solve_handle     - Function handle @(K, F) returning full u_master

if nargin < 4, opts = struct(); end
if ~iscell(bc_list) && isstruct(bc_list)
    bc_list = num2cell(bc_list);
end

n_master = mesh.n_master;
total_dof = 3 * n_master;

K_bc = mesh.K_master;
F_bc = zeros(total_dof, 1);

fixed_dofs_all = [];
prescribed_vals_all = [];

v1 = brep.nodes(brep.elements(:, 1), :);
v2 = brep.nodes(brep.elements(:, 2), :);
v3 = brep.nodes(brep.elements(:, 3), :);
centroids = (v1 + v2 + v3) / 3.0;
n_brep_facets = size(brep.elements, 1);

% Master node coordinates
master_coords = mesh.nodes(mesh.master_node_ids, :);

for k = 1:numel(bc_list)
    bc = bc_list{k};
    bc_type = lower(bc.type);
    
    % Resolve facet filter
    facet_idx = [];
    if isfield(bc, 'filter') && ~isempty(bc.filter)
        if isa(bc.filter, 'function_handle')
            try
                mask = bc.filter(centroids(:,1), centroids(:,2), centroids(:,3));
            catch
                mask = false(n_brep_facets, 1);
                for fi = 1:n_brep_facets
                    mask(fi) = bc.filter(centroids(fi, 1), centroids(fi, 2), centroids(fi, 3));
                end
            end
            facet_idx = find(mask);
        elseif isnumeric(bc.filter) && isvector(bc.filter) && all(bc.filter == floor(bc.filter)) && max(bc.filter) <= n_brep_facets
            facet_idx = bc.filter(:);
        elseif isnumeric(bc.filter) && all(size(bc.filter) == [3, 2])
            % Bounding box filter
            bb = bc.filter;
            mask = (centroids(:,1) >= bb(1,1)-1e-5 & centroids(:,1) <= bb(1,2)+1e-5 & ...
                    centroids(:,2) >= bb(2,1)-1e-5 & centroids(:,2) <= bb(2,2)+1e-5 & ...
                    centroids(:,3) >= bb(3,1)-1e-5 & centroids(:,3) <= bb(3,2)+1e-5);
            facet_idx = find(mask);
        end
    else
        facet_idx = (1:n_brep_facets)';
    end
    
    switch bc_type
        case {'neumann', 'traction', 'pressure'}
            val = bc.value;
            [F_neu, ~] = assemble_neumann_bc_3d(mesh, brep, facet_idx, val);
            F_bc = F_bc + F_neu;
            
        case {'robin', 'spring', 'elastic_foundation'}
            k_s = 0.0;
            if isfield(bc, 'k_spring'), k_s = bc.k_spring; end
            val = [0, 0, 0];
            if isfield(bc, 'value'), val = bc.value; end
            [K_rob, F_rob, ~] = assemble_robin_bc_3d(mesh, brep, facet_idx, k_s, val);
            K_bc = K_bc + K_rob;
            F_bc = F_bc + F_rob;
            
        case {'dirichlet', 'fixed', 'clamp', 'prescribed'}
            method = 'weak'; % default for immersed CAD boundaries
            if isfield(bc, 'method'), method = lower(bc.method); end
            
            val = [0, 0, 0];
            if isfield(bc, 'value'), val = bc.value; end
            if isscalar(val), val = [val, val, val]; end
            
            comps = [1, 1, 1];
            if isfield(bc, 'components'), comps = logical(bc.components); end
            
            if strcmp(method, 'penalty') || strcmp(method, 'weak') || strcmp(method, 'nitsche')
                % Weak penalty/Nitsche Dirichlet formulation on immersed B-Rep boundary:
                pen = 1e7;
                if isfield(bc, 'penalty'), pen = bc.penalty; end
                [K_pen, F_pen, ~] = assemble_robin_bc_3d(mesh, brep, facet_idx, pen, pen * val);
                K_bc = K_bc + K_pen;
                F_bc = F_bc + F_pen;
            else
                % Strong Dirichlet enforcement on master nodes
                matching_nodes = [];
                if isfield(bc, 'node_filter') && isa(bc.node_filter, 'function_handle')
                    % Explicit filter on master node coordinates
                    try
                        n_mask = bc.node_filter(master_coords(:,1), master_coords(:,2), master_coords(:,3));
                    catch
                        n_mask = false(n_master, 1);
                        for mi = 1:n_master
                            n_mask(mi) = bc.node_filter(master_coords(mi,1), master_coords(mi,2), master_coords(mi,3));
                        end
                    end
                    matching_nodes = find(n_mask);
                elseif ~isempty(facet_idx)
                    % Identify leaf elements containing Dirichlet facets
                    elem_set = [];
                    bounds = mesh.bounds;
                    for fi = facet_idx(:)'
                        pt = centroids(fi, :);
                        e_idx = find(pt(1) >= bounds(:,1)-1e-5 & pt(1) <= bounds(:,2)+1e-5 & ...
                                     pt(2) >= bounds(:,3)-1e-5 & pt(2) <= bounds(:,4)+1e-5 & ...
                                     pt(3) >= bounds(:,5)-1e-5 & pt(3) <= bounds(:,6)+1e-5, 1);
                        if ~isempty(e_idx), elem_set = [elem_set, e_idx]; end
                    end
                    elem_set = unique(elem_set);
                    active_nodes = unique(mesh.elem_nodes(elem_set, :));
                    
                    % Find master nodes that influence these nodes via T
                    for an = active_nodes(:)'
                        m_deps = find(mesh.T(an, :) > 0.05);
                        matching_nodes = [matching_nodes, m_deps(:)'];
                    end
                    matching_nodes = unique(matching_nodes);
                end
                
                % Add to fixed master DOFs
                for m_id = matching_nodes(:)'
                    for c = 1:3
                        if comps(c)
                            dof = (m_id - 1) * 3 + c;
                            fixed_dofs_all = [fixed_dofs_all; dof];
                            if isa(val, 'function_handle')
                                pt_m = master_coords(m_id, :);
                                v_eval = val(pt_m(1), pt_m(2), pt_m(3));
                                prescribed_vals_all = [prescribed_vals_all; v_eval(c)];
                            else
                                prescribed_vals_all = [prescribed_vals_all; val(c)];
                            end
                        end
                    end
                end
            end
    end
end

% Remove duplicate fixed DOFs
if ~isempty(fixed_dofs_all)
    [fixed_dofs_all, u_idx] = unique(fixed_dofs_all);
    prescribed_vals_all = prescribed_vals_all(u_idx);
end

free_dofs_all = setdiff((1:total_dof)', fixed_dofs_all);

bc_state.fixed_dofs = fixed_dofs_all;
bc_state.prescribed_vals = prescribed_vals_all;
bc_state.free_dofs = free_dofs_all;
bc_state.K_unreduced = K_bc;
bc_state.F_unreduced = F_bc;

% Provide solver handle that automatically handles strong elimination
bc_state.solve = @(K, F) solve_with_bc(K, F, fixed_dofs_all, prescribed_vals_all, free_dofs_all, total_dof);

end

function u_master = solve_with_bc(K, F, fixed_dofs, prescribed_vals, free_dofs, total_dof)
% Solves system with strongly eliminated DOFs
u_master = zeros(total_dof, 1);
if isempty(fixed_dofs)
    u_master = K \ F;
else
    u_master(fixed_dofs) = prescribed_vals;
    F_mod = F(free_dofs) - K(free_dofs, fixed_dofs) * prescribed_vals;
    u_master(free_dofs) = K(free_dofs, free_dofs) \ F_mod;
end
end
