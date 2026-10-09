% test_gpu_matrixfree_pcg.m
addpath(genpath('C:\Users\dshgu\OneDrive\Documents\geopdes-master'));
addpath('c:\Users\dshgu\OneDrive\Documents\FastFormation');

nelx = 40; nely = 20; p = 3;
pd = cantilever_beam(1.0, 0.5);
md.degree = [p p]; md.regularity = [p-1 p-1]; md.nsub = [nelx nely]; md.nquad = [p+1 p+1];
[geometry, msh, sp] = buildSpaces(pd, md);
[free_dofs, ~] = grab_cantilever_dofs(sp);

F = op_f_v_tp(sp, msh, pd.f);
Fy_tot = abs(sum(F(sp.scalar_spaces{1}.ndof + 1 : end)));
if Fy_tot > 0, F = F / Fy_tot; end

% Precompute element stiffness matrices
sp_col = sp_precompute(sp, msh, 'gradient', true, 'divergence', true);
msh_col = msh_precompute(msh);
l_val = pd.lambda_lame(0, 0) * ones(msh.nqn, msh.nel);
m_val = pd.mu_lame(0, 0) * ones(msh.nqn, msh.nel);
[rows_all, cols_all, vals0_all] = op_su_ev(sp_col, sp_col, msh_col, l_val, m_val);

% Extract per-element connectivity and local matrices
nel = msh.nel;
nsh = sp_col.nsh_max;
rows_e = reshape(rows_all, [nsh, nsh, nel]);
cols_e = reshape(cols_all, [nsh, nsh, nel]);
vals_e = reshape(vals0_all, [nsh, nsh, nel]);

conn_e = squeeze(rows_e(:, 1, :)); % [nsh x nel] global DOF indices for each element

% Test matrix-vector multiplication with random density
xPhys = 0.5 * ones(nelx, nely);
scale = 1e-3 + (1 - 1e-3) * (xPhys(:).^3);

% CPU reference assembly & solve
n_per_el = nsh^2;
K_cpu = sparse(rows_all, cols_all, vals0_all .* repelem(scale, n_per_el), sp.ndof, sp.ndof);
K_cpu = 0.5 * (K_cpu + K_cpu');
u_cpu = zeros(sp.ndof, 1);
u_cpu(free_dofs) = K_cpu(free_dofs, free_dofs) \ F(free_dofs);

% GPU Matrix-Free PCG implementation
conn_gpu = gpuArray(int32(conn_e));
vals_gpu = gpuArray(single(vals_e));
scale_gpu = gpuArray(single(scale));
F_gpu = gpuArray(single(F));
free_mask = false(sp.ndof, 1); free_mask(free_dofs) = true;
free_mask_gpu = gpuArray(free_mask);

% Compute diagonal preconditioner matrix-free
% Each element contributes diag(vals_e(:,:,e)) * scale(e)
diag_e = zeros(nsh, nel, 'single');
for a = 1:nsh
    diag_e(a, :) = squeeze(vals_e(a, a, :))' .* scale';
end
K_diag = accumarray(conn_e(:), diag_e(:), [sp.ndof, 1]);
M_inv_gpu = gpuArray(single(1 ./ max(K_diag, 1e-6)));

% Matvec function on GPU
matvec = @(p) eval_matvec(p, conn_gpu, vals_gpu, scale_gpu, free_mask_gpu, sp.ndof, nsh, nel);

% Custom GPU PCG
tic;
u_mf = zeros(sp.ndof, 1, 'single', 'gpuArray');
r = F_gpu .* free_mask_gpu - matvec(u_mf);
z = M_inv_gpu .* r;
p_vec = z;
rz_old = sum(r .* z);
tol = 1e-5; max_iter = 250;
for iter = 1:max_iter
    Ap = matvec(p_vec);
    alpha = rz_old / sum(p_vec .* Ap);
    u_mf = u_mf + alpha * p_vec;
    r = r - alpha * Ap;
    res_norm = norm(r) / norm(F_gpu(free_dofs));
    if res_norm < tol, break; end
    z = M_inv_gpu .* r;
    rz_new = sum(r .* z);
    beta = rz_new / rz_old;
    p_vec = z + beta * p_vec;
    rz_old = rz_new;
end
wait(gpuDevice);
t_pcg = toc;

u_mf_cpu = gather(double(u_mf));
rel_err = norm(u_mf_cpu(free_dofs) - u_cpu(free_dofs)) / norm(u_cpu(free_dofs));
fprintf('GPU Matrix-Free PCG completed in %d iterations (%.4f s), relative error: %.3e\n', ...
    iter, t_pcg, rel_err);

function y = eval_matvec(p, conn, vals, scale, free_mask, ndof, nsh, nel)
    p_active = p .* free_mask;
    pe = p_active(conn); % [nsh x nel]
    pe_reshaped = reshape(pe, [nsh, 1, nel]);
    ye = pagemtimes(vals, pe_reshaped); % [nsh x 1 x nel]
    ye_scaled = ye .* reshape(scale, [1, 1, nel]);
    y_full = accumarray(conn(:), ye_scaled(:), [ndof, 1]);
    y = y_full .* free_mask;
end
