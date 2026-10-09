function [K_contact, contact_info] = assemble_octree_nitsche_contact(meshA, meshB, brepA, brepB, opts)
% ASSEMBLE_OCTREE_NITSCHE_CONTACT
% Evaluates weak bonded contact / interface coupling between two bodies
% discretized by independent non-conforming adaptive octrees using Nitsche's method.
%
% Inputs:
%   meshA, meshB - Structural octree meshes (from octree_structural_mesh)
%   brepA, brepB - Surface B-Rep meshes (.nodes, .elements)
%   opts         - Struct with options (.gap_tol, .gamma_c)
%
% Outputs:
%   K_contact    - Sparse symmetric matrix of size (3*N_mA + 3*N_mB)^2
%   contact_info - Struct with contact statistics

if nargin < 5, opts = struct(); end
if ~isfield(opts, 'gap_tol'), opts.gap_tol = 5.0; end
if ~isfield(opts, 'gamma_c'), opts.gamma_c = 100.0; end

ndofA = 3 * meshA.n_master;
ndofB = 3 * meshB.n_master;
ndof_tot = ndofA + ndofB;

% Extract surface facets and centroids
vA1 = brepA.nodes(brepA.elements(:,1), :);
vA2 = brepA.nodes(brepA.elements(:,2), :);
vA3 = brepA.nodes(brepA.elements(:,3), :);
centA = (vA1 + vA2 + vA3) / 3;
normA = cross(vA2 - vA1, vA3 - vA1, 2);
areaA = 0.5 * sqrt(sum(normA.^2, 2));

vB1 = brepB.nodes(brepB.elements(:,1), :);
vB2 = brepB.nodes(brepB.elements(:,2), :);
vB3 = brepB.nodes(brepB.elements(:,3), :);
centB = (vB1 + vB2 + vB3) / 3;

if isfield(opts, 'contact_pts') && ~isempty(opts.contact_pts)
    contact_pts = opts.contact_pts;
    contact_areas = opts.contact_areas;
else
    % Find proximity contact pairs
    gap_tol = opts.gap_tol;
    nA = size(centA, 1);
    nB = size(centB, 1);
    
    contact_A_idx = [];
    contact_B_idx = [];
    contact_pts = [];
    contact_areas = [];
    
    min_bB = min(brepB.nodes);
    max_bB = max(brepB.nodes);
    
    chunk_sz = 500;
    for sA = 1:chunk_sz:nA
        eA = min(nA, sA + chunk_sz - 1);
        cA_sub = centA(sA:eA, :);
        
        % Proximity test to bounding footprint of Body B within gap_tol
        in_footprint = (cA_sub(:,1) >= min_bB(1) - gap_tol & cA_sub(:,1) <= max_bB(1) + gap_tol & ...
                        cA_sub(:,2) >= min_bB(2) - gap_tol & cA_sub(:,2) <= max_bB(2) + gap_tol & ...
                        abs(cA_sub(:,3) - min_bB(3)) <= gap_tol);
                    
        if any(in_footprint)
            matched_A = (sA - 1) + find(in_footprint);
            contact_A_idx = [contact_A_idx; matched_A(:)];
            contact_pts = [contact_pts; cA_sub(in_footprint, :)];
            contact_areas = [contact_areas; areaA(matched_A)];
        end
    end
end

n_contact = size(contact_pts, 1);
if n_contact == 0
    fprintf('No contact interface pairs found within gap_tol = %.2f\n', gap_tol);
    K_contact = sparse(ndof_tot, ndof_tot);
    contact_info.n_pairs = 0;
    return;
end

hA = mean(mean(meshA.h_elem));
hB = mean(mean(meshB.h_elem));
h_eff = 0.5 * (hA + hB);
penalty = opts.gamma_c / h_eff;

% Locate containing elements in meshA and meshB
bA = meshA.bounds;
bB = meshB.bounds;

% Reference corners for shape function evaluation
xi_c  = [-1,  1,  1, -1, -1,  1,  1, -1];
eta_c = [-1, -1,  1,  1, -1, -1,  1,  1];
zt_c  = [-1, -1, -1, -1,  1,  1,  1,  1];

% Preallocate triplets for unconstrained contact coupling
% Max 4 blocks: AA, AB, BA, BB, each with 24x24 DOFs
max_triplets = n_contact * 4 * (24 * 24);
row_trip = zeros(max_triplets, 1);
col_trip = zeros(max_triplets, 1);
val_trip = zeros(max_triplets, 1);
ptr = 0;

for c = 1:n_contact
    pt = contact_pts(c, :);
    dA = contact_areas(c);
    
    % Find element in meshA
    inA = find(pt(1) >= bA(:,1)-1e-5 & pt(1) <= bA(:,2)+1e-5 & ...
               pt(2) >= bA(:,3)-1e-5 & pt(2) <= bA(:,4)+1e-5 & ...
               pt(3) >= bA(:,5)-1e-5 & pt(3) <= bA(:,6)+1e-5, 1);
           
    % Find element in meshB
    inB = find(pt(1) >= bB(:,1)-1e-5 & pt(1) <= bB(:,2)+1e-5 & ...
               pt(2) >= bB(:,3)-1e-5 & pt(2) <= bB(:,4)+1e-5 & ...
               pt(3) >= bB(:,5)-1e-5 & pt(3) <= bB(:,6)+1e-5, 1);
           
    if isempty(inA) || isempty(inB)
        continue;
    end
    
    % Evaluate shape functions in element A
    b_elA = bA(inA, :);
    h_elA = b_elA([2 4 6]) - b_elA([1 3 5]);
    c_elA = 0.5 * (b_elA([1 3 5]) + b_elA([2 4 6]));
    xi_A  = max(-1, min(1, 2 * (pt(1) - c_elA(1)) / h_elA(1)));
    eta_A = max(-1, min(1, 2 * (pt(2) - c_elA(2)) / h_elA(2)));
    zt_A  = max(-1, min(1, 2 * (pt(3) - c_elA(3)) / h_elA(3)));
    NA = 0.125 * (1 + xi_c * xi_A) .* (1 + eta_c * eta_A) .* (1 + zt_c * zt_A);
    
    % Evaluate shape functions in element B
    b_elB = bB(inB, :);
    h_elB = b_elB([2 4 6]) - b_elB([1 3 5]);
    c_elB = 0.5 * (b_elB([1 3 5]) + b_elB([2 4 6]));
    xi_B  = max(-1, min(1, 2 * (pt(1) - c_elB(1)) / h_elB(1)));
    eta_B = max(-1, min(1, 2 * (pt(2) - c_elB(2)) / h_elB(2)));
    zt_B  = max(-1, min(1, 2 * (pt(3) - c_elB(3)) / h_elB(3)));
    NB = 0.125 * (1 + xi_c * xi_B) .* (1 + eta_c * eta_B) .* (1 + zt_c * zt_B);
    
    % Degrees of freedom
    nodesA = meshA.elem_nodes(inA, :);
    nodesB = meshB.elem_nodes(inB, :);
    
    dofsA = [(nodesA-1)*3+1; (nodesA-1)*3+2; (nodesA-1)*3+3];
    dofsA = dofsA(:)';
    
    % In unconstrained global assembly, Body B nodes are offset by 3*meshA.n_nodes
    dofsB = 3 * meshA.n_nodes + [(nodesB-1)*3+1; (nodesB-1)*3+2; (nodesB-1)*3+3];
    dofsB = dofsB(:)';
    
    % Interface block matrices for 3 displacement components
    NA_mat = kron(NA, eye(3));
    NB_mat = kron(NB, eye(3));
    
    K_aa = (penalty * dA) * (NA_mat' * NA_mat);
    K_bb = (penalty * dA) * (NB_mat' * NB_mat);
    K_ab = -(penalty * dA) * (NA_mat' * NB_mat);
    K_ba = -(penalty * dA) * (NB_mat' * NA_mat);
    
    % Block AA
    [r_aa, c_aa] = meshgrid(dofsA, dofsA);
    n_ent = 24 * 24;
    row_trip(ptr + (1:n_ent)) = r_aa(:);
    col_trip(ptr + (1:n_ent)) = c_aa(:);
    val_trip(ptr + (1:n_ent)) = K_aa(:);
    ptr = ptr + n_ent;
    
    % Block BB
    [r_bb, c_bb] = meshgrid(dofsB, dofsB);
    row_trip(ptr + (1:n_ent)) = r_bb(:);
    col_trip(ptr + (1:n_ent)) = c_bb(:);
    val_trip(ptr + (1:n_ent)) = K_bb(:);
    ptr = ptr + n_ent;
    
    % Block AB
    [r_ab, c_ab] = meshgrid(dofsA, dofsB);
    row_trip(ptr + (1:n_ent)) = r_ab(:);
    col_trip(ptr + (1:n_ent)) = c_ab(:);
    val_trip(ptr + (1:n_ent)) = K_ab(:);
    ptr = ptr + n_ent;
    
    % Block BA
    [r_ba, c_ba] = meshgrid(dofsB, dofsA);
    row_trip(ptr + (1:n_ent)) = r_ba(:);
    col_trip(ptr + (1:n_ent)) = c_ba(:);
    val_trip(ptr + (1:n_ent)) = K_ba(:);
    ptr = ptr + n_ent;
end

row_trip = row_trip(1:ptr);
col_trip = col_trip(1:ptr);
val_trip = val_trip(1:ptr);

N_uncon_tot = 3 * meshA.n_nodes + 3 * meshB.n_nodes;
K_uncon_contact = sparse(row_trip, col_trip, val_trip, N_uncon_tot, N_uncon_tot);

% Project to master space: T_tot = blkdiag(T_3d_A, T_3d_B)
T_tot = blkdiag(meshA.T_3d, meshB.T_3d);
K_contact = T_tot' * K_uncon_contact * T_tot;
K_contact = 0.5 * (K_contact + K_contact');

contact_info.n_pairs = n_contact;
contact_info.points = contact_pts;
contact_info.penalty = penalty;

fprintf('Assembled Octree Nitsche Contact: %d interface points, matrix size %d x %d (nnz = %d)\n', ...
    n_contact, size(K_contact, 1), size(K_contact, 2), nnz(K_contact));

end
