function [Ke, type_id] = iga_elasticity_element_matrices(sp, lambda, mu)
% IGA_ELASTICITY_ELEMENT_MATRICES  Exact linear-elasticity element stiffness on a box.
%   [Ke, type_id] = iga_elasticity_element_matrices(sp, lambda, mu)
%     sp       space from IGA_SPACE_BOX
%     lambda, mu  constant Lame parameters
%   Returns the distinct element matrices Ke [nsh x nsh x ntypes] and, for each
%   element, its type: the stiffness of element e is Ke(:, :, type_id(e)).
%   Rows are test functions and columns trial functions, both in the local
%   ordering of sp.connectivity(:, e), i.e. the layout of GeoPDEs op_su_ev.
%
%   a(u, v) = int lambda div(u) div(v) + 2 mu eps(u):eps(v). On an affine box the
%   integrals factor into 1D element matrices, so with p+1 Gauss points per
%   direction the result is exact. Elements whose 1D matrices coincide in every
%   direction share one type (interior elements of a uniform grid are one type).

dim = sp.dim;
nsh_sc = sp.nsh_sc;

% 1D element matrices per direction: M{d}(:, :, e, t), t = 00, 10, 01, 11
% (first index = test function, "1" = physical derivative on that side).
M = cell(1, dim);
tid_dir = cell(1, dim);
for d = 1:dim
    u = sp.univ(d);
    Ld = sp.L(d);
    nl = u.degree + 1;
    Md = zeros(nl, nl, u.nel, 4);
    for e = 1:u.nel
        w = u.qw(:, e) * Ld;                         % physical weights
        Nv = u.shape_functions(:, :, e);
        dNv = u.shape_function_gradients(:, :, e) / Ld; % physical derivatives
        Md(:, :, e, 1) = Nv.' * (w .* Nv);
        Md(:, :, e, 2) = dNv.' * (w .* Nv);
        Md(:, :, e, 3) = Nv.' * (w .* dNv);
        Md(:, :, e, 4) = dNv.' * (w .* dNv);
    end
    M{d} = Md;
    % Group 1D elements with identical matrices (relative tolerance 1e-12).
    feat = reshape(permute(Md, [1 2 4 3]), [], u.nel).';
    scale = max(abs(feat(:)));
    [~, first, tid_dir{d}] = unique(round(feat / scale * 1e12), 'rows', 'stable');
    M{d} = Md(:, :, first, :);
end

sub = cell(1, dim);
[sub{:}] = ind2sub(sp.nel_dir, 1:sp.nel);
combo = zeros(sp.nel, dim);
for d = 1:dim
    combo(:, d) = tid_dir{d}(sub{d});
end
[types, ~, type_id] = unique(combo, 'rows', 'stable');
type_id = type_id.';
ntypes = size(types, 1);

Ke = zeros(sp.nsh, sp.nsh, ntypes);
for t = 1:ntypes
    % G{i,j} = int d_i(phi_test) d_j(phi_trial), scalar functions, kron over directions.
    G = cell(dim, dim);
    for i = 1:dim
        for j = 1:dim
            Gij = 1;
            for d = 1:dim
                if d == i && d == j
                    k = 4;
                elseif d == i
                    k = 2;
                elseif d == j
                    k = 3;
                else
                    k = 1;
                end
                Gij = kron(M{d}(:, :, types(t, d), k), Gij);
            end
            G{i, j} = Gij;
        end
    end
    lap = zeros(nsh_sc);
    for k = 1:dim
        lap = lap + G{k, k};
    end
    for ci = 1:dim
        rows = (ci - 1) * nsh_sc + (1:nsh_sc);
        for cj = 1:dim
            cols = (cj - 1) * nsh_sc + (1:nsh_sc);
            blk = lambda * G{ci, cj} + mu * G{cj, ci};
            if ci == cj
                blk = blk + mu * lap;
            end
            Ke(rows, cols, t) = blk;
        end
    end
end
end
