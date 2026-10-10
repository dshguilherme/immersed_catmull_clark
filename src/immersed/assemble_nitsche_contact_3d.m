function [K_contact, F_contact, contact_pairs] = assemble_nitsche_contact_3d(brepA, brepB, spA, spB, grid_boundsA, grid_boundsB, opts)
% ASSEMBLE_NITSCHE_CONTACT_3D
% Evaluates weak contact / interface coupling between two immersed B-Rep bodies (A and B)
% across non-conforming background grids using Nitsche's formulation.
%
% Energy contribution:
%   a_c(u, v) = \int_{\Gamma_c} [ \gamma_c/h (u_A - u_B) \cdot (v_A - v_B) 
%               - \langle \sigma(u) n \rangle \cdot (v_A - v_B) - \langle \sigma(v) n \rangle \cdot (u_A - u_B) ] d\Gamma
%
% Inputs:
%   brepA, brepB - Surface meshes (.nodes, .elements)
%   spA, spB     - Vector spline spaces for body A and B
%   grid_boundsA, grid_boundsB - [3 x 2] bounding boxes
%   opts         - Struct with options (.gap_tol, .gamma_c, .grid_resA, .grid_resB)

if nargin < 7, opts = struct(); end
if ~isfield(opts, 'gap_tol'), opts.gap_tol = 5.0; end % Max proximity gap for contact interface
if ~isfield(opts, 'gamma_c'), opts.gamma_c = 50.0; end % Contact penalty parameter
if ~isfield(opts, 'friction'), opts.friction = 0.0; end % Bonded / frictionless

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

% Vectorized fast nearest-neighbor search without external toolbox dependency
% For candidate contact pairs, filter first by Z proximity
gap_tol = opts.gap_tol;
nA = size(centA, 1);
nB = size(centB, 1);

contact_facetsA = [];
contact_facetsB = [];
contact_gaps = [];

% Block-wise search for efficiency
chunk_sz = 500;
for sA = 1:chunk_sz:nA
    eA = min(nA, sA + chunk_sz - 1);
    cA_sub = centA(sA:eA, :);
    
    % Candidate B facets close in Z
    z_minA = min(cA_sub(:,3)) - gap_tol;
    z_maxA = max(cA_sub(:,3)) + gap_tol;
    candB = find(centB(:,3) >= z_minA & centB(:,3) <= z_maxA);
    if isempty(candB), continue; end
    
    cB_sub = centB(candB, :);
    dx = cA_sub(:,1) - cB_sub(:,1)';
    dy = cA_sub(:,2) - cB_sub(:,2)';
    dz = cA_sub(:,3) - cB_sub(:,3)';
    dist2 = dx.^2 + dy.^2 + dz.^2;
    
    [min_dist2, min_idxB] = min(dist2, [], 2);
    min_dist = sqrt(min_dist2);
    
    match_mask = (min_dist <= gap_tol);
    if any(match_mask)
        matched_A = (sA - 1) + find(match_mask);
        matched_B = candB(min_idxB(match_mask));
        contact_facetsA = [contact_facetsA; matched_A(:)];
        contact_facetsB = [contact_facetsB; matched_B(:)];
        contact_gaps = [contact_gaps; min_dist(match_mask)];
    end
end

n_contact = numel(contact_facetsA);
contact_pairs.facetsA = contact_facetsA;
contact_pairs.facetsB = contact_facetsB;
contact_pairs.gap = contact_gaps;
contact_pairs.n_pairs = n_contact;

ndofA = spA.ndof;
ndofB = spB.ndof;
ndof_tot = ndofA + ndofB;

if n_contact == 0
    fprintf('No contact facets found within gap_tol = %.2f\n', opts.gap_tol);
    K_contact = sparse(ndof_tot, ndof_tot);
    F_contact = zeros(ndof_tot, 1);
    return;
end

fprintf('Identified %d Nitsche contact facet pairs (Mean gap: %.3e)\n', ...
    n_contact, mean(contact_pairs.gap));

% Characteristic element size
hA = mean(diff(grid_boundsA, 1, 2) ./ opts.grid_resA(:));
hB = mean(diff(grid_boundsB, 1, 2) ./ opts.grid_resB(:));
h_contact = 0.5 * (hA + hB);
penalty = opts.gamma_c / h_contact;

% Spline space dimensions
ncpA = spA.ndof_dir;
ndof_scA = spA.ndof_sc;
ncpB = spB.ndof_dir;
ndof_scB = spB.ndof_sc;

xgA = linspace(grid_boundsA(1,1), grid_boundsA(1,2), ncpA(1));
ygA = linspace(grid_boundsA(2,1), grid_boundsA(2,2), ncpA(2));
zgA = linspace(grid_boundsA(3,1), grid_boundsA(3,2), ncpA(3));

xgB = linspace(grid_boundsB(1,1), grid_boundsB(1,2), ncpB(1));
ygB = linspace(grid_boundsB(2,1), grid_boundsB(2,2), ncpB(2));
zgB = linspace(grid_boundsB(3,1), grid_boundsB(3,2), ncpB(3));

% Assemble Nitsche interface penalty: \int \gamma/h (u_A - u_B) \cdot (v_A - v_B) dA
i_list = []; j_list = []; s_list = [];

for k = 1:n_contact
    fA = contact_facetsA(k);
    fB = contact_facetsB(k);
    
    xA = centA(fA, :);
    a_f = areaA(fA);
    
    % Interpolation weights in Body A
    [WA, dofsA] = eval_trilinear_weights(xA, xgA, ygA, zgA, ncpA, ndof_scA);
    % Interpolation weights in Body B (evaluated at the contact interface xA)
    [WB, dofsB] = eval_trilinear_weights(xA, xgB, ygB, zgB, ncpB, ndof_scB);
    dofsB_global = dofsB + ndofA; % Offset for body B DOFs
    
    % Composite interface vector: [uA; uB]
    % (uA - uB) = [WA; -WB] * [uA; uB]
    for comp = 0:2
        dA = dofsA + comp * ndof_scA;
        dB = dofsB_global + comp * ndof_scB;
        
        d_all = [dA; dB];
        w_all = [WA; -WB];
        
        W_outer = (penalty * a_f) * (w_all * w_all');
        [I_mat, J_mat] = ndgrid(d_all, d_all);
        
        i_list = [i_list; I_mat(:)];
        j_list = [j_list; J_mat(:)];
        s_list = [s_list; W_outer(:)];
    end
end

K_contact = sparse(i_list, j_list, s_list, ndof_tot, ndof_tot);
F_contact = zeros(ndof_tot, 1);

end

function [W, cp_indices] = eval_trilinear_weights(x, xg, yg, zg, ncp, ndof_sc)
    xf = x(1); yf = x(2); zf = x(3);
    
    ix = find(xg <= xf, 1, 'last'); if isempty(ix), ix = 1; elseif ix >= ncp(1), ix = ncp(1)-1; end
    iy = find(yg <= yf, 1, 'last'); if isempty(iy), iy = 1; elseif iy >= ncp(2), iy = ncp(2)-1; end
    iz = find(zg <= zf, 1, 'last'); if isempty(iz), iz = 1; elseif iz >= ncp(3), iz = ncp(3)-1; end
    
    [iX, iY, iZ] = ndgrid(ix:ix+1, iy:iy+1, iz:iz+1);
    cp_indices = sub2ind(ncp, iX(:), iY(:), iZ(:));
    
    wx = [(xg(ix+1) - xf), (xf - xg(ix))] / (xg(ix+1) - xg(ix));
    wy = [(yg(iy+1) - yf), (yf - yg(iy))] / (yg(iy+1) - yg(iy));
    wz = [(zg(iz+1) - zf), (zf - zg(iz))] / (zg(iz+1) - zg(iz));
    [Wx, Wy, Wz] = ndgrid(wx, wy, wz);
    W = Wx(:) .* Wy(:) .* Wz(:);
end
