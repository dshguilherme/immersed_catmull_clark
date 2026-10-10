function sol = solve_immersed_iga_3d(brep, opts)
% SOLVE_IMMERSED_IGA_3D
% General immersed Isogeometric Analysis (IGA) 3D elasticity solver
% using Catmull-Clark / uniform B-spline basis functions, cut-cell Weighted
% Quadrature, Ghost Penalty stabilization, and FastFormation GPU Matrix-Free PCG.
%
% Inputs:
%   brep - B-Rep surface mesh (.nodes, .elements)
%   opts - Struct with solver configurations:
%          .grid_res     - [nx, ny, nz] (default: [24, 16, 12])
%          .padding      - Bounding box padding fraction (default: 0.05)
%          .degree       - Spline degree p (default: 2)
%          .E            - Young's modulus (default: 1.0)
%          .nu           - Poisson's ratio (default: 0.3)
%          .gamma_gp     - Ghost penalty parameter (default: 0.05)
%          .gamma_nitsche- Nitsche penalty parameter (default: 20.0)
%          .use_nitsche  - Enforce Dirichlet via weak Nitsche (default: false)
%          .dirichlet_filter - @(centroid) logical filter for Nitsche facets
%          .clamped_face - Background boundary face to clamp ('bottom', 'left', etc.)
%          .load_case    - Load definition struct
%          .pcg_tol      - PCG tolerance (default: 1e-4)
%          .pcg_maxit    - PCG maximum iterations (default: 300)
%          .use_gpu      - Boolean to use GPU acceleration (default: true)
%
% Outputs:
%   sol  - Struct with full solution results and visualization fields

if nargin < 2, opts = struct(); end
if ~isfield(opts, 'grid_res'), opts.grid_res = [24, 16, 12]; end
if ~isfield(opts, 'padding'), opts.padding = 0.05; end
if ~isfield(opts, 'degree'), opts.degree = 2; end
if ~isfield(opts, 'E'), opts.E = 1.0; end
if ~isfield(opts, 'nu'), opts.nu = 0.3; end
if ~isfield(opts, 'gamma_gp'), opts.gamma_gp = 0.05; end
if ~isfield(opts, 'gamma_nitsche'), opts.gamma_nitsche = 20.0; end
if ~isfield(opts, 'use_nitsche'), opts.use_nitsche = false; end
if ~isfield(opts, 'dirichlet_filter'), opts.dirichlet_filter = []; end
if ~isfield(opts, 'clamped_face'), opts.clamped_face = 'bottom'; end
if ~isfield(opts, 'load_case'), opts.load_case = struct(); end
if ~isfield(opts, 'pcg_tol'), opts.pcg_tol = 1e-4; end
if ~isfield(opts, 'pcg_maxit'), opts.pcg_maxit = 300; end
if ~isfield(opts, 'use_gpu'), opts.use_gpu = (gpuDeviceCount > 0); end

% Lamé parameters
E = opts.E; nu = opts.nu;
lambda = (E * nu) / ((1 + nu) * (1 - 2*nu));
mu = E / (2 * (1 + nu));

% 1. Compute Bounding Box
minCoords = min(brep.nodes, [], 1);
maxCoords = max(brep.nodes, [], 1);
pad = opts.padding * (maxCoords - minCoords);
grid_bounds = [minCoords(1)-pad(1), maxCoords(1)+pad(1); ...
               minCoords(2)-pad(2), maxCoords(2)+pad(2); ...
               minCoords(3)-pad(3), maxCoords(3)+pad(3)];

grid_res = opts.grid_res;
nelx = grid_res(1); nely = grid_res(2); nelz = grid_res(3);
nel = nelx * nely * nelz;

Lx = grid_bounds(1,2) - grid_bounds(1,1);
Ly = grid_bounds(2,2) - grid_bounds(2,1);
Lz = grid_bounds(3,2) - grid_bounds(3,1);
hx = Lx / nelx; hy = Ly / nely; hz = Lz / nelz;
h_cell = [hx, hy, hz];

% 2. Build Background Spline Space
p = opts.degree;
sp = iga_space_box(grid_bounds, grid_res, p);
conn_e = sp.connectivity;
nsh = sp.nsh;
ndof = sp.ndof;
ndof_sc = sp.ndof_sc;
ncp_dir = sp.ndof_dir;

% 3. Cut-Cell Quadrature and Classification
[elem_weights, status] = assemble_immersed_element_weights(brep, grid_bounds, grid_res, [4, 4, 4]);

% Minimum stiffness scaling for void background
Emin = 1e-4;
scale = Emin + (1.0 - Emin) * elem_weights(:);

% 4. Ghost Penalty Stabilization
if opts.gamma_gp > 0
    [K_gp, diag_gp] = assemble_ghost_penalty_stabilization(sp, conn_e, status, grid_res, h_cell, opts.gamma_gp);
else
    K_gp = sparse(ndof, ndof);
    diag_gp = zeros(ndof, 1);
end

% 5. Element Stiffness (exact per element type, any degree)
[Ke_types, type_id] = iga_elasticity_element_matrices(sp, lambda, mu);
Ke_types = single(Ke_types);
vals_e = Ke_types(:, :, type_id);

% 6. Boundary Conditions
[i1, i2, i3] = ind2sub(ncp_dir, 1:ndof_sc);
F = zeros(ndof, 1, 'single');
free_mask = true(ndof, 1);

if opts.use_nitsche && ~isempty(opts.dirichlet_filter)
    % Weak Nitsche Dirichlet BC on B-Rep
    [K_nitsche, F_nitsche] = assemble_nitsche_dirichlet_3d(brep, sp, grid_bounds, grid_res, ...
        opts.dirichlet_filter, [0; 0; 0], opts.gamma_nitsche);
    K_gp = K_gp + K_nitsche;
    diag_gp = diag_gp + full(diag(K_nitsche));
    F = F + single(F_nitsche);
else
    % Strong Dirichlet on background face
    switch lower(opts.clamped_face)
        case 'bottom'
            clamped_sc = find(i3 == 1);
        case 'left'
            clamped_sc = find(i1 == 1);
        case 'top'
            clamped_sc = find(i3 == ncp_dir(3));
        otherwise
            clamped_sc = find(i3 == 1);
    end
    clamped_dofs = [clamped_sc, clamped_sc + ndof_sc, clamped_sc + 2*ndof_sc];
    free_mask(clamped_dofs) = false;
end
free_dofs = find(free_mask);

% External Force
if isfield(opts.load_case, 'force_vector')
    F = F + single(opts.load_case.force_vector);
else
    % Default: central load on top boundary surface
    mid_x = round(ncp_dir(1)/2);
    mid_y = round(ncp_dir(2)/2);
    load_sc = find(i3 == ncp_dir(3) & abs(i1 - mid_x) <= 2 & abs(i2 - mid_y) <= 2);
    load_dofs_z = load_sc + 2*ndof_sc;
    F(load_dofs_z) = -1.0 / numel(load_dofs_z);
end

% 7. Solve on GPU or CPU
if opts.use_gpu
    conn_gpu = gpuArray(int32(conn_e));
    vals_gpu = gpuArray(vals_e);
    F_gpu = gpuArray(F);
    scale_gpu = gpuArray(single(scale));
    free_mask_gpu = gpuArray(free_mask);
    K_gp_gpu = gpuArray(K_gp);

    diag_vals = zeros(nsh, nel, 'like', vals_gpu);
    for a = 1:nsh
        diag_vals(a, :) = squeeze(vals_gpu(a, a, :))' .* scale_gpu';
    end
    K_diag_vol = accumarray(conn_gpu(:), diag_vals(:), [ndof, 1]);
    K_diag_total = K_diag_vol + gpuArray(single(diag_gp));
    M_inv = 1 ./ max(K_diag_total, single(1e-6));

    matvec = @(p_in) eval_matvec_3d_gp(p_in, conn_gpu, vals_gpu, scale_gpu, K_gp_gpu, free_mask_gpu, ndof);
    u_state = zeros(ndof, 1, 'single', 'gpuArray');
    r = (F_gpu - matvec(u_state)) .* free_mask_gpu;
    z = M_inv .* r;
    p_vec = z;
    rz_old = sum(r .* z);
    tol = single(opts.pcg_tol);
    norm_f = norm(F_gpu(free_dofs));

    t_pcg = tic;
    for pcg_it = 1:opts.pcg_maxit
        Ap = matvec(p_vec);
        pAp = sum(p_vec .* Ap);
        if abs(pAp) < 1e-12, break; end
        alpha = rz_old / pAp;
        u_state = u_state + alpha * p_vec;
        r = r - alpha * Ap;
        rel_res = norm(r) / norm_f;
        if rel_res < tol, break; end
        z = M_inv .* r;
        rz_new = sum(r .* z);
        p_vec = z + (rz_new / rz_old) * p_vec;
        rz_old = rz_new;
    end
    time_solve = toc(t_pcg);
    u = gather(double(u_state));
    compliance = gather(double(sum(F_gpu .* u_state)));
else
    % CPU Fallback
    matvec = @(p_in) eval_matvec_3d_gp(p_in, conn_e, vals_e, scale, K_gp, free_mask, ndof);
    % Standard CPU PCG
    diag_vals = zeros(nsh, nel, 'single');
    for a = 1:nsh
        diag_vals(a, :) = squeeze(vals_e(a, a, :))' .* scale';
    end
    K_diag_vol = accumarray(conn_e(:), diag_vals(:), [ndof, 1]);
    K_diag_total = K_diag_vol + single(diag_gp);
    M_inv = 1 ./ max(K_diag_total, 1e-6);

    u_state = zeros(ndof, 1, 'single');
    r = (F - matvec(u_state)) .* free_mask;
    z = M_inv .* r;
    p_vec = z;
    rz_old = sum(r .* z);
    tol = single(opts.pcg_tol);
    norm_f = norm(F(free_dofs));

    t_pcg = tic;
    for pcg_it = 1:opts.pcg_maxit
        Ap = matvec(p_vec);
        pAp = sum(p_vec .* Ap);
        if abs(pAp) < 1e-12, break; end
        alpha = rz_old / pAp;
        u_state = u_state + alpha * p_vec;
        r = r - alpha * Ap;
        rel_res = norm(r) / norm_f;
        if rel_res < tol, break; end
        z = M_inv .* r;
        rz_new = sum(r .* z);
        p_vec = z + (rz_new / rz_old) * p_vec;
        rz_old = rz_new;
    end
    time_solve = toc(t_pcg);
    u = double(u_state);
    compliance = double(sum(F .* u_state));
end

% 8. Interpolate Displacement onto B-Rep Surface Nodes
ux = u(1:ndof_sc); uy = u(ndof_sc+1:2*ndof_sc); uz = u(2*ndof_sc+1:3*ndof_sc);
u_mag_cp = sqrt(ux.^2 + uy.^2 + uz.^2);
u_mag_grid = reshape(u_mag_cp, ncp_dir);

xg_vec = linspace(grid_bounds(1,1), grid_bounds(1,2), ncp_dir(1));
yg_vec = linspace(grid_bounds(2,1), grid_bounds(2,2), ncp_dir(2));
zg_vec = linspace(grid_bounds(3,1), grid_bounds(3,2), ncp_dir(3));

F_interp = griddedInterpolant({xg_vec, yg_vec, zg_vec}, u_mag_grid, 'linear', 'nearest');
u_on_brep = F_interp(brep.nodes(:,1), brep.nodes(:,2), brep.nodes(:,3));

% Pack results
sol.u = u;
sol.compliance = compliance;
sol.pcg_iters = pcg_it;
sol.time_solve = time_solve;
sol.elem_weights = elem_weights;
sol.cell_status = status;
sol.grid_bounds = grid_bounds;
sol.grid_res = grid_res;
sol.sp = sp;
sol.conn_e = conn_e;
sol.u_on_brep = u_on_brep;
sol.K_gp = K_gp;

end

function y = eval_matvec_3d_gp(p, conn, vals, scale, K_gp, free_mask, ndof)
    p_act = p .* free_mask;
    pe = p_act(conn);
    [nsh_val, nel_val] = size(pe);
    pe_reshaped = reshape(pe, [nsh_val, 1, nel_val]);
    ye = pagemtimes(vals, pe_reshaped);
    ye_scaled = ye .* reshape(scale, [1, 1, nel_val]);
    y_full = accumarray(conn(:), ye_scaled(:), [ndof, 1]);
    % Add ghost penalty contribution
    y_full = y_full + K_gp * p_act;
    y = y_full .* free_mask;
end
