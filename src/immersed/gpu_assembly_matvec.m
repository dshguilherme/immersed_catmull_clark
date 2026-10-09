function [y, t_gpu] = gpu_assembly_matvec(assembly, p, K_contact, K_bc_extra, opts)
% GPU_ASSEMBLY_MATVEC
% Performs matrix-free or GPU-accelerated matrix-vector multiplication for
% coupled multi-body CAD assemblies with contact interface couplings:
%   y = (K_assembly_bulk + K_contact + K_bc_extra) * p
%
% Inputs:
%   assembly   - Multi-body struct from setup_assembly_3d
%   p          - Input vector [total_dof x 1] (CPU double or gpuArray)
%   K_contact  - (Optional) Contact coupling matrix [total_dof x total_dof]
%   K_bc_extra - (Optional) Robin/penalty BC extra matrix [total_dof x total_dof]
%   opts       - (Optional) Struct:
%                  .use_gpu - Logical (default: auto-detect)
%
% Outputs:
%   y          - Result vector [total_dof x 1]
%   t_gpu      - Wall time (s)

if nargin < 3, K_contact = []; end
if nargin < 4, K_bc_extra = []; end
if nargin < 5, opts = struct(); end

has_gpu = (exist('gpuDeviceCount', 'file') && gpuDeviceCount() > 0);
use_gpu = has_gpu;
if isfield(opts, 'use_gpu'), use_gpu = opts.use_gpu; end

t_start = tic;
total_dof = assembly.total_dof;
n_bodies = assembly.n_bodies;

y = zeros(total_dof, 1);

% 1. Evaluate bulk matrix-free matvec body-by-body
for i = 1:n_bodies
    b = assembly.bodies{i};
    dofs = b.dof_range;
    p_i = p(dofs);
    
    matvec_opts.use_gpu = use_gpu;
    y_i = gpu_octree_matvec(b.mesh, p_i, matvec_opts);
    y(dofs) = y_i;
end

% 2. Add contact coupling if present
if ~isempty(K_contact)
    y = y + K_contact * p;
end

% 3. Add boundary condition extra matrix if present
if ~isempty(K_bc_extra)
    y = y + K_bc_extra * p;
end

t_gpu = toc(t_start);

end
