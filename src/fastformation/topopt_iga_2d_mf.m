function [xPhys, compliance_hist, time_hist] = topopt_iga_2d_mf(nelx, nely, volfrac, penal, rmin, max_iter, degree, precision_type)
% TOPOPT_IGA_2D_MF 2D Isogeometric Topology Optimization via Matrix-Free GPU PCG
%
% Inputs:
%   nelx, nely     - Mesh elements in x and y (default: 80, 40)
%   volfrac        - Target volume fraction (default: 0.5)
%   penal          - SIMP exponent (default: 3.0)
%   rmin           - Filter radius (default: 2.0)
%   max_iter       - Maximum iterations (default: 30)
%   degree         - Polynomial degree p (default: 3)
%   precision_type - 'fp32' (default) or 'fp64'

if nargin < 1 || isempty(nelx), nelx = 80; end
if nargin < 2 || isempty(nely), nely = 40; end
if nargin < 3 || isempty(volfrac), volfrac = 0.5; end
if nargin < 4 || isempty(penal), penal = 3.0; end
if nargin < 5 || isempty(rmin), rmin = 2.0; end
if nargin < 6 || isempty(max_iter), max_iter = 30; end
if nargin < 7 || isempty(degree), degree = 3; end
if nargin < 8 || isempty(precision_type), precision_type = 'fp32'; end

L = 1.0; h = 0.5;
E0 = 1; nu = 0.3;
lambda = nu * E0 / ((1 + nu) * (1 - 2 * nu));
mu = E0 / (2 * (1 + nu));
space = iga_space_box([0 L; 0 h], [nelx, nely], degree);
free_dofs = setdiff(1:space.ndof, space.boundary(1).dofs);
free_mask = false(space.ndof, 1); free_mask(free_dofs) = true;

F = iga_load_vector(space, @(x, y) iga_cantilever_tip_load(x, y, L, h, 'center'));
Fy_tot = abs(sum(F(space.ndof_sc + 1 : end)));
if Fy_tot > 0, F = F / Fy_tot; end

% Precompute element operators
[Ke, type_id] = iga_elasticity_element_matrices(space, lambda, mu);
nel = space.nel;
nsh = space.nsh;
vals_e = Ke(:, :, type_id);
conn_e = space.connectivity; % [nsh x nel]

% Precompute sensitivity filter
[cx, cy] = ndgrid((0.5:nelx)*(L/nelx), (0.5:nely)*(h/nely));
cx = cx(:); cy = cy(:);
r_phys = rmin * mean([L/nelx, h/nely]);
[I, J] = ndgrid(1:nel, 1:nel);
dist2 = (cx(I) - cx(J)).^2 + (cy(I) - cy(J)).^2;
mask_f = dist2 <= (r_phys^2);
H_filter = sparse(I(mask_f), J(mask_f), max(0, r_phys - sqrt(dist2(mask_f))), nel, nel);
H_sum = full(sum(H_filter, 2));

% GPU Data Setup
is_fp32 = strcmpi(precision_type, 'fp32');
if is_fp32
    prec_fn = @single;
else
    prec_fn = @double;
end

conn_gpu = gpuArray(int32(conn_e));
vals_gpu = gpuArray(prec_fn(vals_e));
F_gpu = gpuArray(prec_fn(F));
free_mask_gpu = gpuArray(free_mask);
u_state = zeros(space.ndof, 1, 'like', F_gpu);

xPhys = volfrac * ones(nel, 1);
Emin = 1e-3;
compliance_hist = zeros(max_iter, 1);
time_hist = zeros(max_iter, 1);

for iter = 1:max_iter
    t_it = tic;
    x_old = xPhys;
    scale = Emin + (1 - Emin) * (xPhys.^penal);
    scale_gpu = gpuArray(prec_fn(scale));
    
    % Diagonal preconditioner
    diag_e = zeros(nsh, nel, 'like', scale_gpu);
    for a = 1:nsh
        diag_e(a, :) = squeeze(vals_gpu(a, a, :))' .* scale_gpu';
    end
    K_diag = accumarray(conn_gpu(:), diag_e(:), [space.ndof, 1]);
    M_inv_gpu = 1 ./ max(K_diag, prec_fn(1e-6));
    
    % Matrix-Free PCG with Warm Start
    matvec = @(p) eval_matvec_2d(p, conn_gpu, vals_gpu, scale_gpu, free_mask_gpu, space.ndof, nsh, nel);
    r = (F_gpu - matvec(u_state)) .* free_mask_gpu;
    z = M_inv_gpu .* r;
    p_vec = z;
    rz_old = sum(r .* z);
    tol = prec_fn(1e-4);
    norm_f = norm(F_gpu(free_dofs));
    
    for pcg_it = 1:150
        Ap = matvec(p_vec);
        pAp = sum(p_vec .* Ap);
        if abs(pAp) < 1e-12, break; end
        alpha = rz_old / pAp;
        u_state = u_state + alpha * p_vec;
        r = r - alpha * Ap;
        if norm(r) / norm_f < tol, break; end
        z = M_inv_gpu .* r;
        rz_new = sum(r .* z);
        p_vec = z + (rz_new / rz_old) * p_vec;
        rz_old = rz_new;
    end
    
    % Sensitivity Evaluation on GPU
    ue = u_state(conn_gpu);
    ke_ue = pagemtimes(vals_gpu, reshape(ue, [nsh, 1, nel]));
    Ee = sum(ue .* squeeze(ke_ue), 1)';
    dC_raw = -penal * (1 - Emin) * (xPhys.^(penal - 1)) .* gather(double(Ee));
    c = gather(double(sum(F_gpu .* u_state)));
    
    % Sensitivity Filtering
    dC = (H_filter * (xPhys .* dC_raw)) ./ (H_sum .* max(xPhys, 1e-3));
    
    % OC Bisection Update
    l1 = 0; l2 = max(abs(dC(:))) * 2.0; move = 0.2;
    while (l2 - l1) / (l1 + l2 + 1e-10) > 1e-4
        lmid = 0.5 * (l1 + l2);
        Be = (-dC / lmid).^0.5;
        x_cand = max(0.001, max(x_old - move, min(1.0, min(x_old + move, x_old .* Be))));
        if mean(x_cand) > volfrac, l1 = lmid; else, l2 = lmid; end
    end
    
    xPhys = x_cand;
    change = max(abs(xPhys(:) - x_old(:)));
    compliance_hist(iter) = abs(c);
    time_hist(iter) = toc(t_it);
    
    if change < 0.01 && iter >= 20, break; end
end

compliance_hist = compliance_hist(1:iter);
time_hist = time_hist(1:iter);

end

function y = eval_matvec_2d(p, conn, vals, scale, free_mask, ndof, nsh, nel)
    p_act = p .* free_mask;
    pe = p_act(conn);
    pe_reshaped = reshape(pe, [nsh, 1, nel]);
    ye = pagemtimes(vals, pe_reshaped);
    ye_scaled = ye .* reshape(scale, [1, 1, nel]);
    y_full = accumarray(conn(:), ye_scaled(:), [ndof, 1]);
    y = y_full .* free_mask;
end
