function y = gpu_octree_matvec(mesh, p, opts)
% GPU_OCTREE_MATVEC
% Evaluates the matrix-free action y = K_master * p directly across octree
% levels without forming the global stiffness matrix in RAM or VRAM:
%   y = T_3d' * ( \sum_e L_e' * (w_e * K_e * (L_e * (T_3d * p))) )
%
% Leverages batched BLAS-3 tensor products across octree levels on GPU/CPU.
%
% Inputs:
%   mesh - Structural octree mesh (from octree_structural_mesh)
%   p    - Master displacement vector [3*N_master x 1] (CPU or gpuArray)
%   opts - (Optional) struct with settings:
%            .use_gpu  - Logical (default: true if GPU available)
%            .K_extra  - Optional additional sparse matrix (Dirichlet/Robin/Contact)
%
% Outputs:
%   y    - Resulting force vector [3*N_master x 1] (matches class of p)

if nargin < 3, opts = struct(); end

use_gpu = false;
if isfield(opts, 'use_gpu')
    use_gpu = opts.use_gpu;
elseif isa(p, 'gpuArray') || (exist('gpuDeviceCount', 'file') && gpuDeviceCount() > 0)
    use_gpu = true;
end

% Forward Multi-Point Constraint projection: p_uncon = T_3d * p
p_uncon = mesh.T_3d * p;

n_nodes = mesh.n_nodes;
n_leaves = mesh.n_elements;
levels = mesh.octree.levels;
weights = mesh.weights;
unique_levels = unique(levels);

% Precompute or retrieve element DOFs
if ~isfield(mesh, 'elem_dofs')
    elem_dofs = zeros(n_leaves, 24);
    for a = 1:8
        elem_dofs(:, (a-1)*3 + (1:3)) = [(mesh.elem_nodes(:, a)-1)*3 + 1, ...
                                         (mesh.elem_nodes(:, a)-1)*3 + 2, ...
                                         (mesh.elem_nodes(:, a)-1)*3 + 3];
    end
else
    elem_dofs = mesh.elem_dofs;
end

% Reconstruct or retrieve ke_by_level
if isfield(mesh, 'ke_by_level')
    ke_by_level = mesh.ke_by_level;
else
    % Compute reference element stencils per level
    bounds = mesh.octree.bounds;
    root_h = [bounds(1,2)-bounds(1,1), bounds(1,4)-bounds(1,3), bounds(1,6)-bounds(1,5)] * (2^levels(1));
    gp = [-1/sqrt(3), 1/sqrt(3)];
    gw = [1, 1];
    xi_c  = [-1,  1,  1, -1, -1,  1,  1, -1];
    eta_c = [-1, -1,  1,  1, -1, -1,  1,  1];
    zt_c  = [-1, -1, -1, -1,  1,  1,  1,  1];
    
    ke_by_level = cell(max(levels)+1, 1);
    for ul = unique_levels(:)'
        h_lvl = root_h / (2^ul);
        ke_lvl = zeros(24, 24);
        detJ = prod(h_lvl) / 8;
        for ix = 1:2
            for iy = 1:2
                for iz = 1:2
                    xi = gp(ix); eta = gp(iy); zt = gp(iz);
                    w = gw(ix) * gw(iy) * gw(iz);
                    dNdxi  = (2/h_lvl(1)) * 0.125 * xi_c  .* (1 + eta_c * eta) .* (1 + zt_c * zt);
                    dNdeta = (2/h_lvl(2)) * 0.125 * eta_c .* (1 + xi_c * xi)   .* (1 + zt_c * zt);
                    dNdzt  = (2/h_lvl(3)) * 0.125 * zt_c  .* (1 + xi_c * xi)   .* (1 + eta_c * eta);
                    B = zeros(6, 24);
                    for a = 1:8
                        col = (a-1)*3 + (1:3);
                        B(:, col) = [dNdxi(a), 0, 0;
                                     0, dNdeta(a), 0;
                                     0, 0, dNdzt(a);
                                     0, dNdzt(a), dNdeta(a);
                                     dNdzt(a), 0, dNdxi(a);
                                     dNdeta(a), dNdxi(a), 0];
                    end
                    ke_lvl = ke_lvl + (B' * mesh.C_tensor * B) * (detJ * w);
                end
            end
        end
        ke_by_level{ul+1} = 0.5 * (ke_lvl + ke_lvl');
    end
end

% Batched Level-by-Level Operator Evaluation
y_uncon = zeros(3 * n_nodes, 1, 'like', p_uncon);

for ul = unique_levels(:)'
    idx_lvl = find(levels == ul);
    n_el_lvl = numel(idx_lvl);
    if n_el_lvl == 0, continue; end
    
    ke_lvl = ke_by_level{ul+1};
    if use_gpu && ~isa(ke_lvl, 'gpuArray') && isa(p, 'gpuArray')
        ke_lvl = gpuArray(ke_lvl);
    end
    
    dofs_lvl = elem_dofs(idx_lvl, :); % [n_el_lvl x 24]
    
    % Gather local displacement vectors: U_lvl is [24 x n_el_lvl]
    U_lvl = reshape(p_uncon(dofs_lvl'), 24, n_el_lvl);
    
    % Batched matrix multiplication: Y_lvl = K_lvl * U_lvl [24 x n_el_lvl]
    Y_lvl = ke_lvl * U_lvl;
    
    % Scale by element cut-cell volume weights
    w_lvl = weights(idx_lvl)'; % [1 x n_el_lvl]
    if use_gpu && ~isa(w_lvl, 'gpuArray') && isa(p, 'gpuArray')
        w_lvl = gpuArray(w_lvl);
    end
    F_lvl = Y_lvl .* repmat(w_lvl, 24, 1);
    
    % Scatter accumulate into y_uncon
    flat_dofs = dofs_lvl';
    if use_gpu && isa(p, 'gpuArray')
        % On GPU, use linear accumulation or sparse reduction
        F_flat = F_lvl(:);
        y_uncon = y_uncon + accumarray(flat_dofs(:), F_flat, [3*n_nodes, 1]);
    else
        F_flat = F_lvl(:);
        y_uncon = y_uncon + accumarray(flat_dofs(:), F_flat, [3*n_nodes, 1]);
    end
end

% Transpose Multi-Point Constraint projection: y = T_3d' * y_uncon
y = mesh.T_3d' * y_uncon;

% Add any extra boundary/contact operator contributions if provided
if isfield(opts, 'K_extra') && ~isempty(opts.K_extra)
    y = y + opts.K_extra * p;
end

end
