function C = iga_wq_elasticity_tensor(sp, lambda, mu, grid_size)
% IGA_WQ_ELASTICITY_TENSOR  Parametric elasticity coefficients for WQ on a box.
%   C = iga_wq_elasticity_tensor(sp, lambda, mu, grid_size) returns a cell
%   C{i, j} (i = test component, j = trial component) of arrays of size
%   [grid_size, dim, dim]; entry (..., k1, k2) multiplies
%   d_{xi_k1}(v_i) * d_{xi_k2}(u_j) in the parametric weak form, including the
%   Jacobian determinant. With the affine box map x_d = a_d + L_d xi_d:
%     C{i,j}(k1,k2) = det(J) / (L_k1 L_k2) *
%                     [lambda d_ik1 d_jk2 + mu (d_ij d_k1k2 + d_ik2 d_jk1)]
%   which is constant over the domain.

dim = sp.dim;
detJ = prod(sp.L);
grid_size = grid_size(:).';
C = cell(dim, dim);
for i = 1:dim
    for j = 1:dim
        c = zeros(dim, dim);
        for k1 = 1:dim
            for k2 = 1:dim
                c(k1, k2) = detJ / (sp.L(k1) * sp.L(k2)) * ...
                    (lambda * (i == k1) * (j == k2) + mu * ((i == j) * (k1 == k2) + (i == k2) * (j == k1)));
            end
        end
        C{i, j} = repmat(reshape(c, [ones(1, numel(grid_size)), dim, dim]), [grid_size, 1, 1]);
    end
end
end
