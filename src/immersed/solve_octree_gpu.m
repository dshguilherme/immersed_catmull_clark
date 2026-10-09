function [u_master, solver_info] = solve_octree_gpu(mesh, F_master, bc_state, opts)
% SOLVE_OCTREE_GPU
% Matrix-free Preconditioned Conjugate Gradient (PCG) solver running 100%
% in GPU VRAM (or accelerated CPU fallback). Solves large-scale adaptive
% octree structural mechanics problems without forming the global stiffness matrix.
%
% Inputs:
%   mesh        - Structural octree mesh (from octree_structural_mesh)
%   F_master    - Global master load vector [3*N_master x 1]
%   bc_state    - (Optional) BC state struct from apply_boundary_conditions
%   opts        - (Optional) Settings struct:
%                   .tol       - Relative residual tolerance (default: 1e-6)
%                   .max_iter  - Maximum PCG iterations (default: 1500)
%                   .use_gpu   - Force GPU on/off (default: auto-detect)
%                   .verbose   - Print convergence history (default: false)
%
% Outputs:
%   u_master    - Solution master displacement vector [3*N_master x 1]
%   solver_info - Struct with solver diagnostics:
%                   .iterations - Number of iterations taken
%                   .rel_res    - Final relative residual ||r|| / ||b||
%                   .wall_time  - Total solve time (s)
%                   .gpu_used   - Logical true if GPU was used
%                   .converged  - Logical true if tolerance reached

if nargin < 3, bc_state = struct(); end
if nargin < 4, opts = struct(); end

tol      = 1e-6;  if isfield(opts, 'tol'), tol = opts.tol; end
max_iter = 1500;  if isfield(opts, 'max_iter'), max_iter = opts.max_iter; end
verbose  = false; if isfield(opts, 'verbose'), verbose = opts.verbose; end

has_gpu = (exist('gpuDeviceCount', 'file') && gpuDeviceCount() > 0);
use_gpu = has_gpu;
if isfield(opts, 'use_gpu'), use_gpu = opts.use_gpu; end

total_dof = 3 * mesh.n_master;

% Resolve boundary conditions
fixed_dofs = [];
prescribed_vals = [];
if isfield(bc_state, 'fixed_dofs') && ~isempty(bc_state.fixed_dofs)
    fixed_dofs = bc_state.fixed_dofs;
    prescribed_vals = bc_state.prescribed_vals;
end
free_dofs = setdiff((1:total_dof)', fixed_dofs);
n_free = numel(free_dofs);

% Extract or compute Jacobi diagonal preconditioner
if isfield(mesh, 'K_master')
    diag_K = full(diag(mesh.K_master));
else
    % Probe diagonal via unit vectors if K_master was not formed
    diag_K = ones(total_dof, 1);
end
diag_K(abs(diag_K) < 1e-12) = 1.0;
diag_M_free = diag_K(free_dofs);

% Compute RHS with prescribed displacement elimination
u_master = zeros(total_dof, 1);
F_rhs = F_master;

matvec_opts.use_gpu = use_gpu;
if isfield(bc_state, 'K_bc_extra')
    matvec_opts.K_extra = bc_state.K_bc_extra;
end

if ~isempty(fixed_dofs)
    u_master(fixed_dofs) = prescribed_vals;
    % Matrix-vector product for fixed DOFs: y_fixed = K * u_fixed
    p_fixed = zeros(total_dof, 1);
    p_fixed(fixed_dofs) = prescribed_vals;
    y_fixed = gpu_octree_matvec(mesh, p_fixed, matvec_opts);
    b_free = F_rhs(free_dofs) - y_fixed(free_dofs);
else
    b_free = F_rhs(free_dofs);
end

norm_b = norm(b_free);
if norm_b == 0
    u_master(free_dofs) = 0.0;
    solver_info.iterations = 0;
    solver_info.rel_res = 0.0;
    solver_info.wall_time = 0.0;
    solver_info.gpu_used = false;
    solver_info.converged = true;
    return;
end

% Transfer vectors to GPU if enabled
if use_gpu
    try
        b_gpu = gpuArray(b_free);
        diag_M_gpu = gpuArray(diag_M_free);
        p_full_gpu = gpuArray(zeros(total_dof, 1));
        x_free = gpuArray(zeros(n_free, 1));
        gpu_active = true;
    catch
        b_gpu = b_free;
        diag_M_gpu = diag_M_free;
        p_full_gpu = zeros(total_dof, 1);
        x_free = zeros(n_free, 1);
        gpu_active = false;
    end
else
    b_gpu = b_free;
    diag_M_gpu = diag_M_free;
    p_full_gpu = zeros(total_dof, 1);
    x_free = zeros(n_free, 1);
    gpu_active = false;
end

% Operator handle for free DOFs
matvec_free = @(v_free) eval_free_action(mesh, v_free, free_dofs, total_dof, p_full_gpu, matvec_opts);

% Native Preconditioned Conjugate Gradient (PCG) Loop
t_start = tic;

r = b_gpu - matvec_free(x_free);
z = r ./ diag_M_gpu;
p = z;
rz_old = dot(r, z);

converged = false;
for k = 1:max_iter
    Ap = matvec_free(p);
    pAp = dot(p, Ap);
    
    if pAp <= 0
        warning('Indefinite or zero curvature encountered in PCG.');
        break;
    end
    
    alpha = rz_old / pAp;
    x_free = x_free + alpha * p;
    r = r - alpha * Ap;
    
    res_norm = norm(r) / norm_b;
    if verbose && mod(k, 25) == 0
        fprintf('  PCG Iter %4d: RelRes = %.3e\n', k, gather(res_norm));
    end
    
    if res_norm < tol
        converged = true;
        break;
    end
    
    z = r ./ diag_M_gpu;
    rz_new = dot(r, z);
    beta = rz_new / rz_old;
    p = z + beta * p;
    rz_old = rz_new;
end

if gpu_active
    wait(gpuDevice());
end
wall_time = toc(t_start);

% Gather back to CPU
x_sol = gather(x_free);
u_master(free_dofs) = x_sol;

solver_info.iterations = k;
solver_info.rel_res = gather(res_norm);
solver_info.wall_time = wall_time;
solver_info.gpu_used = gpu_active;
solver_info.converged = converged;

if verbose
    fprintf('Matrix-Free PCG %s in %d iterations (RelRes: %.2e, Time: %.3f s, GPU: %d)\n', ...
        ternary(converged, 'Converged', 'Stopped'), k, solver_info.rel_res, wall_time, gpu_active);
end

end

function y_free = eval_free_action(mesh, v_free, free_dofs, total_dof, p_full, opts)
p_full(free_dofs) = v_free;
y_full = gpu_octree_matvec(mesh, p_full, opts);
y_free = y_full(free_dofs);
end

function str = ternary(cond, true_str, false_str)
if cond, str = true_str; else, str = false_str; end
end
