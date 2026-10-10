function results = topopt_immersed_iga_3d(brep_input, opts)
% TOPOPT_IMMERSED_IGA_3D
% 3D Immersed Topology Optimization on arbitrary CAD B-Rep models (STEP / MSH / VTU)
% using Catmull-Clark / uniform B-splines, Cut-Cell Weighted Quadrature,
% Ghost Penalty stabilization, and FastFormation GPU Matrix-Free PCG.
%
% Inputs:
%   brep_input - Either a B-Rep struct (.nodes, .elements) or path to a STEP/MSH/VTU file
%   opts       - Configuration struct:
%                .grid_res      - [nx, ny, nz] (default: [28, 20, 16])
%                .volfrac       - Target volume fraction in CAD domain (default: 0.35)
%                .max_iter      - Number of iterations (default: 30)
%                .penal         - SIMP penalty exponent (default: 3.0)
%                .rmin          - Filter radius in element units (default: 1.8)
%                .gamma_gp      - Ghost Penalty stabilization (default: 0.05)
%                .pcg_tol       - PCG solver tolerance (default: 1e-4)
%                .pcg_maxit     - Max PCG iterations per step (default: 150)
%                .output_fig    - Path to output figure
%
% Outputs:
%   results    - Struct with history of compliance, density distribution, and runtime.

if nargin < 2, opts = struct(); end
if ~isfield(opts, 'grid_res'), opts.grid_res = [28, 20, 16]; end
if ~isfield(opts, 'volfrac'), opts.volfrac = 0.35; end
if ~isfield(opts, 'max_iter'), opts.max_iter = 30; end
if ~isfield(opts, 'penal'), opts.penal = 3.0; end
if ~isfield(opts, 'rmin'), opts.rmin = 1.8; end
if ~isfield(opts, 'gamma_gp'), opts.gamma_gp = 0.05; end
if ~isfield(opts, 'pcg_tol'), opts.pcg_tol = 1e-4; end
if ~isfield(opts, 'pcg_maxit'), opts.pcg_maxit = 150; end
if ~isfield(opts, 'output_fig'), opts.output_fig = 'figures/fig_topopt_nist_immersed_3d.png'; end

% 1. Load B-Rep CAD model
if ischar(brep_input) || isstring(brep_input)
    fprintf('Loading CAD geometry from %s...\n', brep_input);
    brep = importBRep(brep_input);
else
    brep = brep_input;
end

% Bounding box with 5% margin
minCoords = min(brep.nodes, [], 1);
maxCoords = max(brep.nodes, [], 1);
pad = 0.05 * (maxCoords - minCoords);
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

fprintf('\n========================================================================\n');
fprintf('  IMMERSED TOPOLOGY OPTIMIZATION ON CAD B-REP (FASTFORMATION GPU)\n');
fprintf('========================================================================\n');
fprintf('Background Mesh: %dx%dx%d (%d elements)\n', nelx, nely, nelz, nel);
fprintf('Target Volume Fraction: %.2f | SIMP Penalty: %.1f | Max Iterations: %d\n', ...
    opts.volfrac, opts.penal, opts.max_iter);

% 2. Cut-Cell Quadrature & Domain Classification
[elem_weights, status] = assemble_immersed_element_weights(brep, grid_bounds, grid_res, [4, 4, 4]);
w_e = elem_weights(:);

% Design domain elements: cells that contain at least 5% CAD volume
active_mask = (w_e >= 0.05);
active_indices = find(active_mask);
n_active = numel(active_indices);
V_cad_total = sum(w_e(active_mask));
fprintf('Active Design Domain: %d / %d elements (CAD Volume: %.4e)\n', ...
    n_active, nel, V_cad_total);

% 3. Background Spline Space & FastFormation Operator Template
p = 2; % Quadratic Catmull-Clark / uniform B-spline
E0 = 1.0; nu = 0.3;
lambda = (E0 * nu) / ((1 + nu) * (1 - 2*nu));
mu = E0 / (2 * (1 + nu));

sp_f = iga_space_box(grid_bounds, grid_res, p);
conn_e = sp_f.connectivity;
nsh = sp_f.nsh;
ndof = sp_f.ndof;
ndof_sc = sp_f.ndof_sc;
ncp_dir = sp_f.ndof_dir;

[Ke_types, type_id] = iga_elasticity_element_matrices(sp_f, lambda, mu);
Ke_types = single(Ke_types);
vals_e = Ke_types(:, :, type_id);

% 4. Ghost Penalty Stabilization on Active Cut Faces
if opts.gamma_gp > 0
    [K_gp, diag_gp] = assemble_ghost_penalty_stabilization(sp_f, conn_e, status, grid_res, h_cell, opts.gamma_gp);
else
    K_gp = sparse(ndof, ndof);
    diag_gp = zeros(ndof, 1);
end

% 5. Boundary Conditions and Loads
% Clamp bottom face (Z = zmin)
[i1, i2, i3] = ind2sub(ncp_dir, 1:ndof_sc);
clamped_sc = find(i3 == 1);
clamped_dofs = [clamped_sc, clamped_sc + ndof_sc, clamped_sc + 2*ndof_sc];
free_mask = true(ndof, 1);
free_mask(clamped_dofs) = false;
free_dofs = find(free_mask);

% Downward traction load on top face center region
mid_x = round(ncp_dir(1)/2);
mid_y = round(ncp_dir(2)/2);
load_sc = find(i3 == ncp_dir(3) & abs(i1 - mid_x) <= 2 & abs(i2 - mid_y) <= 2);
F = zeros(ndof, 1, 'single');
load_dofs_z = load_sc + 2*ndof_sc;
F(load_dofs_z) = -1.0 / numel(load_dofs_z);

% 6. Build Sensitivity Filter Matrix
r_phys = opts.rmin * mean(h_cell);
fprintf('Building Cartesian sensitivity filter (radius: %.2f)...\n', r_phys);
[H_filter, H_sum] = build_cartesian_filter_3d(grid_res, h_cell, r_phys, active_mask);

% 7. Initialize Design Variables
xPhys = zeros(nel, 1, 'single');
xPhys(active_mask) = single(opts.volfrac);
x_old = xPhys;

% Transfer static structures to GPU
conn_gpu = gpuArray(int32(conn_e));
vals_gpu = gpuArray(vals_e);
F_gpu = gpuArray(F);
free_mask_gpu = gpuArray(free_mask);
K_gp_gpu = gpuArray(K_gp);
diag_gp_gpu = gpuArray(single(diag_gp));
H_filter_gpu = gpuArray(single(H_filter));
H_sum_gpu = gpuArray(single(H_sum));
w_gpu = gpuArray(single(w_e));

u_state = zeros(ndof, 1, 'single', 'gpuArray');

compliance_hist = zeros(opts.max_iter, 1);
vol_hist = zeros(opts.max_iter, 1);
delta_hist = zeros(opts.max_iter, 1);
time_hist = zeros(opts.max_iter, 1);

Emin = single(1e-4);
penal = single(opts.penal);

fprintf('\n%-6s %-16s %-14s %-14s %-10s %-10s\n', ...
    'Iter', 'Compliance J', 'CAD VolFrac', 'Delta_rho_max', 'PCG It', 'Time [s]');
fprintf('%s\n', repmat('-', 1, 74));

t_opt_start = tic;

for iter = 1:opts.max_iter
    t_it = tic;
    x_old = xPhys;
    
    % Compute physical element stiffness scale: w_e * (Emin + (1 - Emin) * rho^p)
    scale_elem = w_gpu .* (Emin + (1.0 - Emin) * (gpuArray(xPhys).^penal));
    % Background void elements outside CAD have minimum stiffness
    scale_elem(~active_mask) = Emin;
    
    % Diagonal Jacobi preconditioner
    diag_e = zeros(nsh, nel, 'like', scale_elem);
    for a = 1:nsh
        diag_e(a, :) = squeeze(vals_gpu(a, a, :))' .* scale_elem';
    end
    K_diag_vol = accumarray(conn_gpu(:), diag_e(:), [ndof, 1]);
    K_diag_total = K_diag_vol + diag_gp_gpu;
    M_inv = 1 ./ max(K_diag_total, single(1e-6));
    
    % Matrix-Free PCG Solver with Warm Start
    matvec = @(p_in) eval_matvec_3d_gp(p_in, conn_gpu, vals_gpu, scale_elem, K_gp_gpu, free_mask_gpu, ndof);
    r = (F_gpu - matvec(u_state)) .* free_mask_gpu;
    z = M_inv .* r;
    p_vec = z;
    rz_old = sum(r .* z);
    norm_f = norm(F_gpu(free_dofs));
    
    for pcg_it = 1:opts.pcg_maxit
        Ap = matvec(p_vec);
        pAp = sum(p_vec .* Ap);
        if abs(pAp) < 1e-12, break; end
        alpha = rz_old / pAp;
        u_state = u_state + alpha * p_vec;
        r = r - alpha * Ap;
        if norm(r) / norm_f < opts.pcg_tol, break; end
        z = M_inv .* r;
        rz_new = sum(r .* z);
        p_vec = z + (rz_new / rz_old) * p_vec;
        rz_old = rz_new;
    end
    
    % Evaluate Strain Energy on GPU
    ue = u_state(conn_gpu);
    ke_ue = pagemtimes(vals_gpu, reshape(ue, [nsh, 1, nel]));
    Ee = sum(ue .* squeeze(ke_ue), 1)'; % [nel x 1] strain energy
    
    % Sensitivities: dC_raw = - w_e * p * (1 - Emin) * rho^(p-1) * Ee
    dC_raw = - w_gpu .* (penal * (1.0 - Emin) * (gpuArray(xPhys).^(penal - 1.0)) .* Ee);
    
    % Filter sensitivities over design domain
    dC_filtered = (H_filter_gpu * (gpuArray(xPhys) .* dC_raw)) ./ (H_sum_gpu .* max(gpuArray(xPhys), single(1e-3)));
    
    dC_cpu = gather(dC_filtered);
    x_old_cpu = xPhys;
    
    % Immersed Optimality Criteria (OC) Bisection Update
    l1 = single(0);
    l2 = single(max(abs(dC_cpu(active_mask))) * 2.0);
    move = single(0.2);
    
    target_vol = single(opts.volfrac * V_cad_total);
    w_active_cpu = w_e(active_mask);
    
    while (l2 - l1) / (l1 + l2 + 1e-10) > 1e-4
        lmid = 0.5 * (l1 + l2);
        Be = (-dC_cpu(active_mask) / lmid).^0.5;
        x_cand = max(single(0.001), max(x_old_cpu(active_mask) - move, ...
                 min(single(1.0), min(x_old_cpu(active_mask) + move, x_old_cpu(active_mask) .* Be))));
        current_vol = sum(w_active_cpu .* x_cand);
        if current_vol > target_vol
            l1 = lmid;
        else
            l2 = lmid;
        end
    end
    
    xPhys = zeros(nel, 1, 'single');
    xPhys(active_mask) = x_cand;
    
    delta_rho = max(abs(xPhys(active_mask) - x_old_cpu(active_mask)));
    c_val = gather(double(sum(F_gpu .* u_state)));
    v_frac = gather(double(sum(w_e(active_mask) .* xPhys(active_mask)) / V_cad_total));
    t_step = toc(t_it);
    
    compliance_hist(iter) = c_val;
    vol_hist(iter) = v_frac;
    delta_hist(iter) = delta_rho;
    time_hist(iter) = t_step;
    
    fprintf('%-6d %-16.4e %-14.3f %-14.4f %-10d %-10.2f\n', ...
        iter, c_val, v_frac, delta_rho, pcg_it, t_step);
end

t_total = toc(t_opt_start);
fprintf('========================================================================\n');
fprintf('Immersed TopOpt Finished in %.2f s (Avg %.2f s/iter)\n', t_total, t_total / opts.max_iter);
fprintf('Initial Compliance: %.4e | Final Compliance: %.4e (Reduction: %.1f%%)\n', ...
    compliance_hist(1), compliance_hist(end), (1 - compliance_hist(end)/compliance_hist(1))*100);

% 8. Render & Save High-Quality Visualization Figure
figDir = fileparts(opts.output_fig);
if ~isempty(figDir) && ~exist(figDir, 'dir'), mkdir(figDir); end

fig = figure('Color', 'w', 'Position', [100, 100, 1200, 520]);

% Left: 3D Immersed Optimal Topology inside CAD Shell
subplot(1, 2, 1);
% 1. Outer CAD B-Rep semi-transparent shell
patch('Faces', brep.elements, 'Vertices', brep.nodes, ...
      'FaceColor', [0.75 0.75 0.8], 'EdgeColor', [0.4 0.4 0.4], ...
      'FaceAlpha', 0.15, 'EdgeAlpha', 0.2);
hold on;

% 2. Isosurface of optimized material inside the CAD domain
x_grid = reshape(xPhys, [nelx, nely, nelz]);
% Permute to [ny, nx, nz] to match MATLAB meshgrid convention
x_perm = permute(x_grid, [2, 1, 3]);

gx = linspace(grid_bounds(1,1), grid_bounds(1,2), nelx);
gy = linspace(grid_bounds(2,1), grid_bounds(2,2), nely);
gz = linspace(grid_bounds(3,1), grid_bounds(3,2), nelz);
[Xg, Yg, Zg] = meshgrid(gx, gy, gz);

iso_val = 0.45;
p_iso = patch(isosurface(Xg, Yg, Zg, x_perm, iso_val));
isonormals(Xg, Yg, Zg, x_perm, p_iso);
set(p_iso, 'FaceColor', [0.15 0.45 0.85], 'EdgeColor', 'none');

axis equal tight; grid on; box on; view(45, 30);
set(gca, 'Color', 'w', 'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2], 'ZColor', [0.2 0.2 0.2]);
camlight('headlight'); camlight('right'); lighting gouraud;
xlabel('X', 'FontWeight', 'bold', 'Color', 'k');
ylabel('Y', 'FontWeight', 'bold', 'Color', 'k');
zlabel('Z', 'FontWeight', 'bold', 'Color', 'k');
title(sprintf('NIST CAD Immersed Topology (\\rho \\geq %.2f, V_f = %.2f)', iso_val, opts.volfrac), ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');

% Right: Convergence Curves
subplot(1, 2, 2);
set(gca, 'Color', 'w');
yyaxis left;
plot(1:opts.max_iter, compliance_hist, 'b-o', 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'b');
ylabel('Compliance J = F^T U', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'b');
set(gca, 'YColor', 'b');

yyaxis right;
plot(1:opts.max_iter, delta_hist, 'r--s', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'r');
ylabel('Max Density Change \Delta\rho_{max}', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r');
set(gca, 'YColor', 'r');

grid on; box on;
set(gca, 'XColor', [0.2 0.2 0.2], 'GridColor', [0.8 0.8 0.8], 'GridAlpha', 0.6);
xlabel('Iteration', 'FontWeight', 'bold', 'Color', 'k');
title('Immersed TopOpt Convergence History', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');

out_fig_path = char(opts.output_fig);
[out_dir, ~, ~] = fileparts(out_fig_path);
if ~isempty(out_dir) && ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

exportgraphics(fig, out_fig_path, 'Resolution', 200);
fprintf('Exported figure: %s\n', out_fig_path);
close(fig);

% Return structured results
results.compliance_hist = compliance_hist;
results.vol_hist = vol_hist;
results.delta_hist = delta_hist;
results.time_hist = time_hist;
results.xPhys = xPhys;
results.w_e = w_e;
results.grid_bounds = grid_bounds;
results.grid_res = grid_res;
results.active_mask = active_mask;

end

function y = eval_matvec_3d_gp(p, conn, vals, scale, K_gp, free_mask, ndof)
    p_act = p .* free_mask;
    pe = p_act(conn);
    [nsh_val, nel_val] = size(pe);
    pe_reshaped = reshape(pe, [nsh_val, 1, nel_val]);
    ye = pagemtimes(vals, pe_reshaped);
    ye_scaled = ye .* reshape(scale, [1, 1, nel_val]);
    y_full = accumarray(conn(:), ye_scaled(:), [ndof, 1]);
    y_full = y_full + K_gp * p_act;
    y = y_full .* free_mask;
end
