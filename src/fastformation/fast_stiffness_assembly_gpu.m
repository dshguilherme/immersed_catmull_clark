function [K, t_breakdown] = fast_stiffness_assembly_gpu(space, YOUNG, POISSON, xPhys, penal, Emin, symmetrize)
% FAST_STIFFNESS_ASSEMBLY_GPU GPU-accelerated matrix formation for IGA Elasticity
% utilizing NVIDIA CUDA hardware via MATLAB gpuArray and batched operations.
%
% Inputs:
%   space       - displacement space on a box (see IGA_SPACE_BOX)
%   YOUNG       - Young's modulus (E0)
%   POISSON     - Poisson's ratio (nu)
%   xPhys       - (Optional) Element density matrix
%   penal       - (Optional) SIMP exponent (default: 3)
%   Emin        - (Optional) Void modulus (default: 1e-9)
%   symmetrize  - (Optional) Boolean flag (default: true)

if nargin < 4 || isempty(xPhys), xPhys = []; end
if nargin < 5 || isempty(penal), penal = 3; end
if nargin < 6 || isempty(Emin), Emin = 1e-9; end
if nargin < 7 || isempty(symmetrize), symmetrize = true; end

t_start = tic;

%% 1. 1D Setup on CPU
t_prep = tic;
nsd = space.dim;
for idim = 1:nsd
    Quad_rules(idim) = iga_wq_rules_1d(space.knots{idim}, space.degree(idim));
    Connectivity(idim).neighbors = Quad_rules(idim).neighbors;
    Connectivity(idim).num_neigh = cellfun(@numel, Connectivity(idim).neighbors);
    qn{idim} = Quad_rules(idim).all_points';
    for ii = 1:space.ndof_dir(idim)
        BSval{idim, ii} = Quad_rules(idim).B(Quad_rules(idim).ind_points{ii}, Connectivity(idim).neighbors{ii}).';
        BSder{idim, ii} = Quad_rules(idim).dB(Quad_rules(idim).ind_points{ii}, Connectivity(idim).neighbors{ii}).';
    end
end
t_breakdown.prep = toc(t_prep);

%% 2. Metric and Constitutive Tensor on Device
t_coeff = tic;
aux_size = cellfun(@numel, qn);

lambda = YOUNG * POISSON / ((1 + POISSON) * (1 - 2 * POISSON));
mu = YOUNG / (2 * (1 + POISSON));
C = iga_wq_elasticity_tensor(space, lambda, mu, aux_size);

if ~isempty(xPhys)
    e_idx = cell(1, nsd);
    for idim = 1:nsd
        e_idx{idim} = discretize(qn{idim}, space.breaks{idim});
    end
    if nsd == 2
        rho_q = xPhys(e_idx{1}, e_idx{2});
    elseif nsd == 3
        rho_q = xPhys(e_idx{1}, e_idx{2}, e_idx{3});
    end
    SIMP_factor = Emin + (rho_q.^penal) * (1 - Emin);
    for i = 1:nsd
        for j = 1:nsd
            for k1 = 1:nsd
                for k2 = 1:nsd
                    if nsd == 2
                        C{i, j}(:, :, k1, k2) = C{i, j}(:, :, k1, k2) .* SIMP_factor;
                    else
                        C{i, j}(:, :, :, k1, k2) = C{i, j}(:, :, :, k1, k2) .* SIMP_factor;
                    end
                end
            end
        end
    end
end

% Transfer metric arrays to GPU
C_gpu = cell(nsd, nsd);
for i = 1:nsd
    for j = 1:nsd
        C_gpu{i, j} = gpuArray(C{i, j});
    end
end
t_breakdown.coeff = toc(t_coeff);

%% 3. GPU Sum Factorization and Contractions
t_sumfact = tic;
N_dof = space.ndof_sc;
n_size = space.ndof_dir;

indices = cell(1, nsd);
[indices{:}] = ind2sub(n_size, 1:N_dof);
indices = cell2mat(indices);
indices = reshape(indices, [N_dof, nsd]);

n_index = zeros(1, nsd);
for ll = 1:nsd
    n_index(ll) = prod(n_size(1:ll-1));
end

nonzeros = prod(arrayfun(@(i) sum(Connectivity(i).num_neigh), 1:nsd));
rows = zeros(1, nonzeros);
cols = zeros(1, nonzeros);

val = cell(nsd, nsd);
for i = 1:nsd
    for j = 1:nsd
        val{i, j} = gpuArray.zeros(1, nonzeros);
    end
end

% Precompute 1D B-spline matrices on GPU
for idim = 1:nsd
    for ii = 1:n_size(idim)
        n_neigh = Connectivity(idim).num_neigh(ii);
        Bval_gpu{idim, ii} = gpuArray(BSval{idim, ii}(1:n_neigh, :));
        Bder_gpu{idim, ii} = gpuArray(BSder{idim, ii}(1:n_neigh, :));
        Q00_gpu{idim, ii} = gpuArray(Quad_rules(idim).quad_weights_00{ii});
        Q10_gpu{idim, ii} = gpuArray(Quad_rules(idim).quad_weights_10{ii});
        Q01_gpu{idim, ii} = gpuArray(Quad_rules(idim).quad_weights_01{ii});
        Q11_gpu{idim, ii} = gpuArray(Quad_rules(idim).quad_weights_11{ii});
    end
end

ncounter = 0;
points = cell(1, nsd);
j_act = cell(1, nsd);
len_j_act = zeros(1, nsd);

for ii = 1:N_dof
    ind = indices(ii, :);
    for ll = 1:nsd
        points{ll} = Quad_rules(ll).ind_points{ind(ll)};
        j_act{ll} = Connectivity(ll).neighbors{ind(ll)};
        len_j_act(ll) = length(j_act{ll});
    end
    i_nonzeros = prod(len_j_act);
    range_idx = ncounter + 1 : ncounter + i_nonzeros;

    for k1 = 1:nsd
        for k2 = 1:nsd
            C_blocks = cell(nsd, nsd);
            for i = 1:nsd
                for j = 1:nsd
                    if nsd == 2
                        C_blocks{i, j} = C_gpu{i, j}(points{1}, points{2}, k1, k2);
                    else
                        C_blocks{i, j} = C_gpu{i, j}(points{1}, points{2}, points{3}, k1, k2);
                    end
                end
            end

            for ll = nsd:-1:1
                if (k1 == k2 && ll == k1)
                    Q_dev = Q11_gpu{ll, ind(ll)};
                    B_dev = Bder_gpu{ll, ind(ll)};
                elseif (k1 ~= k2 && ll == k1)
                    Q_dev = Q10_gpu{ll, ind(ll)};
                    B_dev = Bval_gpu{ll, ind(ll)};
                elseif (k1 ~= k2 && ll == k2)
                    Q_dev = Q01_gpu{ll, ind(ll)};
                    B_dev = Bder_gpu{ll, ind(ll)};
                else
                    Q_dev = Q00_gpu{ll, ind(ll)};
                    B_dev = Bval_gpu{ll, ind(ll)};
                end
                B_weighted = bsxfun(@times, Q_dev, B_dev);

                if nsd == 2
                    if ll == 2
                        for i = 1:nsd
                            for j = 1:nsd
                                C_blocks{i, j} = C_blocks{i, j} * B_weighted';
                            end
                        end
                    elseif ll == 1
                        for i = 1:nsd
                            for j = 1:nsd
                                C_blocks{i, j} = B_weighted * C_blocks{i, j};
                            end
                        end
                    end
                end
            end

            for i = 1:nsd
                for j = 1:nsd
                    val{i, j}(range_idx) = val{i, j}(range_idx) + C_blocks{i, j}(:)';
                end
            end
        end
    end

    rows(range_idx) = ii;

    i_col = zeros(nsd, i_nonzeros);
    for ll = 1:nsd
        rep = len_j_act; rep(ll) = 1;
        perm = ones(1, nsd); perm(ll) = len_j_act(ll);
        ap = repmat(reshape(j_act{ll}', perm), rep);
        i_col(ll, :) = ap(:)';
    end
    cols(range_idx) = 1 + n_index * (i_col - 1);

    ncounter = ncounter + i_nonzeros;
end

% Single gather operation for all values
for i = 1:nsd
    for j = 1:nsd
        val{i, j} = gather(val{i, j});
    end
end
wait(gpuDevice);
t_breakdown.sumfact = toc(t_sumfact);

%% 4. Sparse Matrix Assembly & Symmetrization
t_sparse = tic;
K_blocks = cell(nsd, nsd);
for i = 1:nsd
    for j = 1:nsd
        K_blocks{i, j} = sparse(rows, cols, val{i, j}, N_dof, N_dof);
    end
end

if nsd == 2
    K = [K_blocks{1, 1}, K_blocks{1, 2}; ...
         K_blocks{2, 1}, K_blocks{2, 2}];
elseif nsd == 3
    K = [K_blocks{1, 1}, K_blocks{1, 2}, K_blocks{1, 3}; ...
         K_blocks{2, 1}, K_blocks{2, 2}, K_blocks{2, 3}; ...
         K_blocks{3, 1}, K_blocks{3, 2}, K_blocks{3, 3}];
end

if symmetrize
    K = 0.5 * (K + K.');
end
t_breakdown.sparse = toc(t_sparse);
t_breakdown.total = toc(t_start);
end
