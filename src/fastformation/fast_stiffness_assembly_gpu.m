function [K, t_breakdown] = fast_stiffness_assembly_gpu(msh, space, geometry, YOUNG, POISSON, xPhys, penal, Emin, symmetrize)
% FAST_STIFFNESS_ASSEMBLY_GPU GPU-accelerated matrix formation for IGA Elasticity
% utilizing NVIDIA CUDA hardware via MATLAB gpuArray and batched operations.
%
% Inputs:
%   msh         - GeoPDEs mesh structure
%   space       - GeoPDEs vector space structure
%   geometry    - GeoPDEs geometry structure
%   YOUNG       - Young's modulus (E0)
%   POISSON     - Poisson's ratio (nu)
%   xPhys       - (Optional) Element density matrix
%   penal       - (Optional) SIMP exponent (default: 3)
%   Emin        - (Optional) Void modulus (default: 1e-9)
%   symmetrize  - (Optional) Boolean flag (default: true)

if nargin < 6 || isempty(xPhys), xPhys = []; end
if nargin < 7 || isempty(penal), penal = 3; end
if nargin < 8 || isempty(Emin), Emin = 1e-9; end
if nargin < 9 || isempty(symmetrize), symmetrize = true; end

t_start = tic;

%% 1. 1D Setup on CPU
t_prep = tic;
nsd = msh.ndim;
for idim = 1:nsd
    sp1d = space.scalar_spaces{1}.sp_univ(idim);
    Connectivity(idim).neighbors = cellfun(@(x) unique(sp1d.connectivity(:, x)).', sp1d.supp, 'UniformOutput', false);
    Connectivity(idim).num_neigh = cellfun(@numel, Connectivity(idim).neighbors);
    Quad_rules(idim) = quadrule_stiff_fast(sp1d);
    brk{idim} = [space.scalar_spaces{1}.knots{idim}(1), space.scalar_spaces{1}.knots{idim}(end)];
    qn{idim} = Quad_rules(idim).all_points';
end

new_msh = msh_cartesian(brk, qn, [], geometry);
space_wq = space.constructor(new_msh);

for idim = 1:nsd
    sp1d = space_wq.scalar_spaces{1}.sp_univ(idim);
    for ii = 1:sp1d.ndof
        BSval{idim, ii} = sp1d.shape_functions(Quad_rules(idim).ind_points{ii}, Connectivity(idim).neighbors{ii}).';
        BSder{idim, ii} = sp1d.shape_function_gradients(Quad_rules(idim).ind_points{ii}, Connectivity(idim).neighbors{ii}).';
    end
end
t_breakdown.prep = toc(t_prep);

%% 2. Metric and Constitutive Tensor on Device
t_coeff = tic;
aux_size = cellfun(@numel, qn);
jac = msh.map_der(qn);

if nsd == 3
    E = zeros(6, 3, 3);
    E(:, :, 1) = [1 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 1; 0 1 0];
    E(:, :, 2) = [0 0 0; 0 1 0; 0 0 0; 0 0 1; 0 0 0; 1 0 0];
    E(:, :, 3) = [0 0 0; 0 0 0; 0 0 1; 0 1 0; 0 0 1; 0 0 0];
else
    E = zeros(3, 2, 2);
    E(:, :, 1) = [1 0; 0 0; 0 1];
    E(:, :, 2) = [0 0; 0 1; 1 0];
end

lambda = YOUNG * POISSON / ((1 + POISSON) * (1 - 2 * POISSON));
mu = YOUNG / (2 * (1 + POISSON));
total_size = nchoosek(nsd + 2 - 1, 2);
small_size = total_size - nsd;
D1 = lambda * ones(nsd) + 2 * mu * eye(nsd);
D2 = mu * eye(small_size);
D0 = blkdiag(D1, D2);

C = C_ijkl(E, jac, D0, nsd, aux_size);

if ~isempty(xPhys)
    e_idx = cell(1, nsd);
    for idim = 1:nsd
        knots_u = unique(space.scalar_spaces{1}.knots{idim});
        e_idx{idim} = discretize(qn{idim}, knots_u);
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
N_dof = space.scalar_spaces{1}.ndof;
n_size = space.scalar_spaces{1}.ndof_dir;

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
