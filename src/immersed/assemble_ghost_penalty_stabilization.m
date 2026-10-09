function [K_gp, diag_gp] = assemble_ghost_penalty_stabilization(sp, conn_e, status, grid_res, h_cell, gamma_gp)
% ASSEMBLE_GHOST_PENALTY_STABILIZATION
% Assembles face-based ghost penalty stabilization across interior facets of
% cut cells to eliminate small-cut ill-conditioning and ensure bounded condition
% numbers for the GPU PCG solver.
%
% Inputs:
%   sp         - Spline space structure (scalar or vector)
%   conn_e     - Connectivity matrix [nsh x nel]
%   status     - Background element classification [nx, ny, nz] (1: in, 0: out, -1: cut)
%   grid_res   - [nx, ny, nz]
%   h_cell     - [hx, hy, hz] element dimensions
%   gamma_gp   - Ghost penalty parameter (default: 0.05)
%
% Outputs:
%   K_gp       - Sparse stabilization matrix [ndof x ndof]
%   diag_gp    - Vector [ndof x 1] diagonal entries for Jacobi preconditioning

if nargin < 6 || isempty(gamma_gp), gamma_gp = 0.05; end

nx = grid_res(1); ny = grid_res(2); nz = grid_res(3);
nel = nx * ny * nz;
ndof = sp.ndof;
nsh = size(conn_e, 1);

status_lin = status(:);
is_active = (status_lin == 1 | status_lin == -1);
is_cut = (status_lin == -1);

% Identify interior faces shared by at least one cut cell and an active neighbor
faces_e1 = [];
faces_e2 = [];

% 1. X-normal faces (between (i,j,k) and (i+1,j,k))
for i = 1:(nx-1)
    for j = 1:ny
        for k = 1:nz
            e1 = sub2ind([nx, ny, nz], i, j, k);
            e2 = sub2ind([nx, ny, nz], i+1, j, k);
            if is_active(e1) && is_active(e2) && (is_cut(e1) || is_cut(e2))
                faces_e1 = [faces_e1; e1];
                faces_e2 = [faces_e2; e2];
            end
        end
    end
end

% 2. Y-normal faces
for i = 1:nx
    for j = 1:(ny-1)
        for k = 1:nz
            e1 = sub2ind([nx, ny, nz], i, j, k);
            e2 = sub2ind([nx, ny, nz], i, j+1, k);
            if is_active(e1) && is_active(e2) && (is_cut(e1) || is_cut(e2))
                faces_e1 = [faces_e1; e1];
                faces_e2 = [faces_e2; e2];
            end
        end
    end
end

% 3. Z-normal faces
for i = 1:nx
    for j = 1:ny
        for k = 1:(nz-1)
            e1 = sub2ind([nx, ny, nz], i, j, k);
            e2 = sub2ind([nx, ny, nz], i, j, k+1);
            if is_active(e1) && is_active(e2) && (is_cut(e1) || is_cut(e2))
                faces_e1 = [faces_e1; e1];
                faces_e2 = [faces_e2; e2];
            end
        end
    end
end

n_faces = numel(faces_e1);
fprintf('Stabilizing %d active cut-cell interior faces via Ghost Penalty...\n', n_faces);

% Stabilizing jump in higher-order normal derivatives
% For Catmull-Clark cubic/quadratic B-splines, penalty on DOFs across the face
h_char = mean(h_cell);
alpha_gp = gamma_gp * h_char;

i_gp = []; j_gp = []; s_gp = [];

for f = 1:n_faces
    e1 = faces_e1(f);
    e2 = faces_e2(f);
    dofs1 = conn_e(:, e1);
    dofs2 = conn_e(:, e2);
    
    % Apply stabilization component-wise (X, Y, Z displacement fields)
    nsh_sc = nsh / 3;
    for c = 1:3
        idx_c = (c - 1) * nsh_sc + (1:nsh_sc);
        dofs1_c = dofs1(idx_c);
        dofs2_c = dofs2(idx_c);
        
        % Discrete jump operator across facet:
        % Penalize difference between control points on cut element and its neighbor
        % For each shared/paired control point, penalize (u_1 - u_2)^2:
        diff_dofs1 = setdiff(dofs1_c, dofs2_c);
        diff_dofs2 = setdiff(dofs2_c, dofs1_c);
        
        % 1. Common control point higher-order variation:
        common_c = intersect(dofs1_c, dofs2_c);
        if ~isempty(common_c)
            n_c = numel(common_c);
            I_mat = repmat(common_c, n_c, 1);
            J_mat = repmat(common_c', 1, n_c);
            S_mat = (alpha_gp / n_c) * (eye(n_c) - (1/n_c) * ones(n_c));
            i_gp = [i_gp; I_mat(:)];
            j_gp = [j_gp; J_mat(:)];
            s_gp = [s_gp; S_mat(:)];
        end
        
        % 2. Jump coupling between unshared/adjacent boundary control points:
        % Binds the cut cell's unshared boundary DOFs directly to the neighbor DOFs:
        n_p = min(numel(diff_dofs1), numel(diff_dofs2));
        if n_p > 0
            p1 = diff_dofs1(1:n_p);
            p2 = diff_dofs2(1:n_p);
            % Penalty: alpha_gp * (u_p1 - u_p2)^2
            i_gp = [i_gp; p1; p2; p1; p2];
            j_gp = [j_gp; p1; p2; p2; p1];
            s_gp = [s_gp; alpha_gp*ones(n_p,1); alpha_gp*ones(n_p,1); -alpha_gp*ones(n_p,1); -alpha_gp*ones(n_p,1)];
        end
    end
end

K_gp = sparse(i_gp, j_gp, s_gp, ndof, ndof);
diag_gp = full(diag(K_gp));

end
