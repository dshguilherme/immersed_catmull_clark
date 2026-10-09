function assembly = setup_assembly_3d(bodies, opts)
% SETUP_ASSEMBLY_3D
% Configures a multi-body CAD assembly of N solid bodies, generates
% independent non-conforming background octrees with scale-adapted resolutions,
% builds structural meshes with MPC hanging-node constraints, and detects
% candidate contact / coupling interface pairs via bounding volume hierarchies.
%
% Inputs:
%   bodies - Cell array of structs, each containing:
%              .brep            - CAD B-Rep surface mesh (.nodes, .elements)
%              .name            - (Optional) Body label/name
%              .material        - (Optional) Struct with .E, .nu
%              .grid_res        - (Optional) Root octree resolution [nx ny nz]
%              .max_level       - (Optional) Max octree refinement depth (default: 1)
%              .refine_criteria - (Optional) Function handle @(cell_bounds)
%   opts   - (Optional) Global options struct:
%              .gap_tol         - Contact proximity tolerance (default: auto)
%              .margin_ratio    - Octree bounding box margin ratio (default: 0.1)
%              .default_E       - Young's modulus fallback (default: 1e5)
%              .default_nu      - Poisson's ratio fallback (default: 0.3)
%
% Outputs:
%   assembly - Struct containing:
%              .bodies          - Array of body structs with .mesh, .dof_range, .ndof
%              .n_bodies        - Total number of bodies
%              .total_dof       - Total master DOFs across the assembly
%              .K_assembly      - Block-diagonal uncoupled master stiffness matrix
%              .interfaces      - Array of detected contact interface pairs
%              .gap_tol         - Proximity tolerance used

if nargin < 2, opts = struct(); end
if ~iscell(bodies) && isstruct(bodies)
    bodies = num2cell(bodies);
end

n_bodies = numel(bodies);
if n_bodies == 0
    error('bodies list must contain at least one body.');
end

margin_ratio = 0.1;
if isfield(opts, 'margin_ratio'), margin_ratio = opts.margin_ratio; end
default_E  = 1e5;  if isfield(opts, 'default_E'), default_E = opts.default_E; end
default_nu = 0.3;  if isfield(opts, 'default_nu'), default_nu = opts.default_nu; end

assembly.bodies = cell(n_bodies, 1);
assembly.n_bodies = n_bodies;

dof_cursor = 0;
K_blocks = cell(n_bodies, 1);

fprintf('=== Initializing Assembly with %d Bodies ===\n', n_bodies);

for i = 1:n_bodies
    b = bodies{i};
    if ~isfield(b, 'name') || isempty(b.name)
        b.name = sprintf('Body_%d', i);
    end
    
    brep = b.brep;
    pts = brep.nodes;
    p_min = min(pts, [], 1);
    p_max = max(pts, [], 1);
    p_span = max(p_max - p_min, 1e-3);
    
    % Expand bounding box with safety margin
    margin = margin_ratio * p_span;
    bb = [p_min(1) - margin(1), p_max(1) + margin(1);
          p_min(2) - margin(2), p_max(2) + margin(2);
          p_min(3) - margin(3), p_max(3) + margin(3)];
          
    grid_res = [2, 2, 2];
    if isfield(b, 'grid_res') && ~isempty(b.grid_res)
        grid_res = b.grid_res;
    end
    
    max_level = 1;
    if isfield(b, 'max_level') && ~isempty(b.max_level)
        max_level = b.max_level;
    end
    
    refine_crit = [];
    if isfield(b, 'refine_criteria'), refine_crit = b.refine_criteria; end
    
    % Build independent adaptive octree
    oct = octree_mesh_3d(bb, grid_res, max_level, refine_crit);
    oct = balance_octree_3d(oct);
    
    % Build structural mechanics mesh with MPC hanging nodes
    mat_opts.E  = default_E;
    mat_opts.nu = default_nu;
    if isfield(b, 'material') && ~isempty(b.material)
        if isfield(b.material, 'E'),  mat_opts.E  = b.material.E;  end
        if isfield(b.material, 'nu'), mat_opts.nu = b.material.nu; end
    end
    
    s_mesh = octree_structural_mesh(oct, brep, mat_opts);
    
    ndof_i = 3 * s_mesh.n_master;
    dof_range = (dof_cursor + 1):(dof_cursor + ndof_i);
    dof_cursor = dof_cursor + ndof_i;
    
    b.mesh = s_mesh;
    b.bbox = bb;
    b.ndof = ndof_i;
    b.dof_range = dof_range;
    
    K_blocks{i} = s_mesh.K_master;
    assembly.bodies{i} = b;
    
    fprintf('  [%s]: %d elements, %d master nodes, %d DOFs\n', ...
        b.name, s_mesh.n_elements, s_mesh.n_master, ndof_i);
end

assembly.total_dof = dof_cursor;
assembly.K_assembly = blkdiag(K_blocks{:});

% Contact Proximity Detection
if isfield(opts, 'gap_tol') && ~isempty(opts.gap_tol)
    gap_tol = opts.gap_tol;
else
    % Auto-estimate gap tolerance as 5% of smallest characteristic body size
    min_size = Inf;
    for i = 1:n_bodies
        bb = assembly.bodies{i}.bbox;
        diag_len = norm(bb(:,2) - bb(:,1));
        min_size = min(min_size, diag_len);
    end
    gap_tol = 0.05 * min_size;
end
assembly.gap_tol = gap_tol;

interfaces = {};
pair_count = 0;

for i = 1:n_bodies
    for j = (i+1):n_bodies
        bb_i = assembly.bodies{i}.bbox;
        bb_j = assembly.bodies{j}.bbox;
        
        % Check bounding box overlap with gap tolerance
        overlap = (bb_i(1,1) <= bb_j(1,2) + gap_tol) && (bb_i(1,2) >= bb_j(1,1) - gap_tol) && ...
                  (bb_i(2,1) <= bb_j(2,2) + gap_tol) && (bb_i(2,2) >= bb_j(2,1) - gap_tol) && ...
                  (bb_i(3,1) <= bb_j(3,2) + gap_tol) && (bb_i(3,2) >= bb_j(3,1) - gap_tol);
              
        if ~overlap, continue; end
        
        % Detailed surface-to-surface projection proximity pairing
        brep_i = assembly.bodies{i}.brep;
        brep_j = assembly.bodies{j}.brep;
        
        v1_i = brep_i.nodes(brep_i.elements(:,1), :);
        v2_i = brep_i.nodes(brep_i.elements(:,2), :);
        v3_i = brep_i.nodes(brep_i.elements(:,3), :);
        n_i = cross(v2_i - v1_i, v3_i - v1_i, 2);
        a_i = 0.5 * sqrt(sum(n_i.^2, 2));
        un_i = n_i ./ max(2 * a_i, 1e-12);
        
        v1_j = brep_j.nodes(brep_j.elements(:,1), :);
        v2_j = brep_j.nodes(brep_j.elements(:,2), :);
        v3_j = brep_j.nodes(brep_j.elements(:,3), :);
        c_j = (v1_j + v2_j + v3_j) / 3.0;
        n_j = cross(v2_j - v1_j, v3_j - v1_j, 2);
        a_j = 0.5 * sqrt(sum(n_j.^2, 2));
        un_j = n_j ./ max(2 * a_j, 1e-12);
        
        contact_pts_A = [];
        contact_pts_B = [];
        contact_areas = [];
        contact_normals = [];
        facets_i = [];
        facets_j = [];
        
        % For each facet on Body j, test projection onto Body i facets
        for fj = 1:size(c_j, 1)
            pj = c_j(fj, :);
            nj = un_j(fj, :);
            
            % Find candidate opposing facets on Body i
            for fi = 1:size(v1_i, 1)
                ni = un_i(fi, :);
                if dot(ni, nj) > -0.2, continue; end % Must oppose
                
                % Normal distance from pj to plane of facet fi
                A = v1_i(fi, :); B = v2_i(fi, :); C = v3_i(fi, :);
                d_norm = abs(dot(pj - A, ni));
                if d_norm > gap_tol, continue; end
                
                % Project pj onto plane of fi
                p_proj = pj - dot(pj - A, ni) * ni;
                
                % Barycentric coordinates of p_proj in triangle ABC
                u = B - A; v = C - A; w = p_proj - A;
                duu = dot(u, u); duv = dot(u, v); dvv = dot(v, v);
                dwu = dot(w, u); dwv = dot(w, v);
                denom = duu * dvv - duv * duv;
                if abs(denom) < 1e-12, continue; end
                
                lam_B = (dvv * dwu - duv * dwv) / denom;
                lam_C = (duu * dwv - duv * dwu) / denom;
                lam_A = 1.0 - lam_B - lam_C;
                
                % Check if projection lies within triangle (with slight tolerance)
                tol_bary = -0.05;
                if (lam_A >= tol_bary) && (lam_B >= tol_bary) && (lam_C >= tol_bary)
                    contact_pts_A = [contact_pts_A; p_proj];
                    contact_pts_B = [contact_pts_B; pj];
                    contact_areas = [contact_areas; a_j(fj)];
                    contact_normals = [contact_normals; ni]; % Outward normal from i to j
                    facets_i = [facets_i; fi];
                    facets_j = [facets_j; fj];
                    break; % Matched best facet
                end
            end
        end
        
        if ~isempty(contact_pts_A)
            pair_count = pair_count + 1;
            inter.body_A = i;
            inter.body_B = j;
            inter.n_pairs = size(contact_pts_A, 1);
            inter.pts_A = contact_pts_A;
            inter.pts_B = contact_pts_B;
            inter.areas = contact_areas;
            inter.normals = contact_normals;
            inter.facets_A = facets_i;
            inter.facets_B = facets_j;
            
            interfaces{pair_count} = inter;
            fprintf('  Interface detected between [%s] and [%s]: %d contact pairs\n', ...
                assembly.bodies{i}.name, assembly.bodies{j}.name, inter.n_pairs);
        end
    end
end

assembly.interfaces = interfaces;
fprintf('=== Assembly Configured: %d Total Master DOFs, %d Contact Interfaces ===\n', ...
    assembly.total_dof, numel(interfaces));

end
