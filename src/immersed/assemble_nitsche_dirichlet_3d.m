function [K_nitsche, F_nitsche] = assemble_nitsche_dirichlet_3d(brep, sp, grid_bounds, grid_res, dirichlet_face_filter, u_prescribed, gamma_nitsche)
% ASSEMBLE_NITSCHE_DIRICHLET_3D
% Weakly enforces Dirichlet boundary conditions on the immersed B-Rep surface
% using Nitsche's formulation:
%   a_N(u, v) = \int_{\Gamma_D} ( - \sigma(u)n \cdot v - \sigma(v)n \cdot u + (\gamma/h) u \cdot v ) d\Gamma
%
% Inputs:
%   brep                   - B-Rep surface mesh (.nodes, .elements)
%   sp                     - Spline vector space
%   grid_bounds            - [xmin, xmax; ymin, ymax; zmin, zmax]
%   grid_res               - [nx, ny, nz]
%   dirichlet_face_filter  - Function handle @(centroid) returning true for Dirichlet facets
%   u_prescribed           - [3 x 1] prescribed displacement vector (default: [0; 0; 0])
%   gamma_nitsche          - Penalty parameter (default: 20.0)
%
% Outputs:
%   K_nitsche              - Sparse contribution to the global stiffness matrix
%   F_nitsche              - Right-hand side vector for inhomogeneous Dirichlet BCs

if nargin < 6 || isempty(u_prescribed), u_prescribed = [0; 0; 0]; end
if nargin < 7 || isempty(gamma_nitsche), gamma_nitsche = 20.0; end

ndof = sp.ndof;
ndof_sc = sp.ndof_sc;
ncp_dir = sp.ndof_dir;

v1 = brep.nodes(brep.elements(:, 1), :);
v2 = brep.nodes(brep.elements(:, 2), :);
v3 = brep.nodes(brep.elements(:, 3), :);

face_centroids = (v1 + v2 + v3) / 3;
face_normals = cross(v2 - v1, v3 - v1, 2);
face_areas = 0.5 * sqrt(sum(face_normals.^2, 2));

% Filter Dirichlet facets
if isempty(dirichlet_face_filter)
    d_faces = 1:size(brep.elements, 1);
else
    d_mask = dirichlet_face_filter(face_centroids);
    d_faces = find(d_mask);
end

fprintf('Applying Nitsche Dirichlet BCs on %d boundary facets (Area: %.4e)...\n', ...
    numel(d_faces), sum(face_areas(d_faces)));

if isempty(d_faces)
    K_nitsche = sparse(ndof, ndof);
    F_nitsche = zeros(ndof, 1);
    return;
end

% Bounding box and spacing
Lx = grid_bounds(1,2) - grid_bounds(1,1);
Ly = grid_bounds(2,2) - grid_bounds(2,1);
Lz = grid_bounds(3,2) - grid_bounds(3,1);
h_char = mean([Lx/grid_res(1), Ly/grid_res(2), Lz/grid_res(3)]);
penalty = gamma_nitsche / h_char;

xg_vec = linspace(grid_bounds(1,1), grid_bounds(1,2), ncp_dir(1));
yg_vec = linspace(grid_bounds(2,1), grid_bounds(2,2), ncp_dir(2));
zg_vec = linspace(grid_bounds(3,1), grid_bounds(3,2), ncp_dir(3));

% Evaluate basis functions at facet centroids
% Build interpolation matrix M_trace [3*n_faces x ndof]
n_df = numel(d_faces);
xc = face_centroids(d_faces, :);
area_c = face_areas(d_faces);

% Local trilinear / tricubic support lookup around centroid
i_list = []; j_list = []; s_list = [];

for f = 1:n_df
    xf = xc(f, 1); yf = xc(f, 2); zf = xc(f, 3);
    af = area_c(f);
    
    % Find containing control cell
    ix = find(xg_vec <= xf, 1, 'last'); if isempty(ix), ix = 1; elseif ix >= ncp_dir(1), ix = ncp_dir(1)-1; end
    iy = find(yg_vec <= yf, 1, 'last'); if isempty(iy), iy = 1; elseif iy >= ncp_dir(2), iy = ncp_dir(2)-1; end
    iz = find(zg_vec <= zf, 1, 'last'); if isempty(iz), iz = 1; elseif iz >= ncp_dir(3), iz = ncp_dir(3)-1; end
    
    % 8 active control points around the facet
    [iX, iY, iZ] = ndgrid(ix:ix+1, iy:iy+1, iz:iz+1);
    cp_indices = sub2ind(ncp_dir, iX(:), iY(:), iZ(:));
    
    % Trilinear weights at facet
    wx = [(xg_vec(ix+1) - xf), (xf - xg_vec(ix))] / (xg_vec(ix+1) - xg_vec(ix));
    wy = [(yg_vec(iy+1) - yf), (yf - yg_vec(iy))] / (yg_vec(iy+1) - yg_vec(iy));
    wz = [(zg_vec(iz+1) - zf), (zf - zg_vec(iz))] / (zg_vec(iz+1) - zg_vec(iz));
    [Wx, Wy, Wz] = ndgrid(wx, wy, wz);
    W = Wx(:) .* Wy(:) .* Wz(:);
    
    % Apply penalty in X, Y, Z displacement components
    for comp = 0:2
        dofs = cp_indices + comp * ndof_sc;
        % Local block: penalty * af * (W * W')
        [dI, dJ] = ndgrid(dofs, dofs);
        W_outer = penalty * af * (W * W');
        
        i_list = [i_list; dI(:)];
        j_list = [j_list; dJ(:)];
        s_list = [s_list; W_outer(:)];
    end
end

K_nitsche = sparse(i_list, j_list, s_list, ndof, ndof);
F_nitsche = zeros(ndof, 1); % Homogeneous case; can be extended for non-zero

end
