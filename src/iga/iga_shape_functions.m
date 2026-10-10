function [Nq, dNq] = iga_shape_functions(sp)
% IGA_SHAPE_FUNCTIONS  Scalar basis values and physical gradients at Gauss points.
%   [Nq, dNq] = iga_shape_functions(sp)
%     Nq   [nqn x nsh_sc x nel]
%     dNq  [dim x nqn x nsh_sc x nel]
%   Local function order is that of sp.conn_sc. Memory grows with nel * nqn *
%   nsh_sc, so use this for moderate meshes (sensitivities, loads), not for the
%   large matrix-free solves.

dim = sp.dim;
sub = cell(1, dim);
[sub{:}] = ind2sub(sp.nel_dir, 1:sp.nel);
qloc = cell(1, dim);
[qloc{:}] = ind2sub(sp.nquad, (1:sp.nqn).');
floc = cell(1, dim);
[floc{:}] = ind2sub(sp.nsh_dir, 1:sp.nsh_sc);

Nq = ones(sp.nqn, sp.nsh_sc, sp.nel);
want_grad = nargout > 1;
if want_grad
    dNq = ones(dim, sp.nqn, sp.nsh_sc, sp.nel);
end
for d = 1:dim
    u = sp.univ(d);
    V = u.shape_functions(qloc{d}, floc{d}, :);                    % [nqn x nsh_sc x nel_d]
    V = V(:, :, sub{d});
    if want_grad
        D = u.shape_function_gradients(qloc{d}, floc{d}, :) / sp.L(d);
        D = D(:, :, sub{d});
        for k = 1:dim
            if k == d
                dNq(k, :, :, :) = reshape(dNq(k, :, :, :), sp.nqn, sp.nsh_sc, sp.nel) .* D;
            else
                dNq(k, :, :, :) = reshape(dNq(k, :, :, :), sp.nqn, sp.nsh_sc, sp.nel) .* V;
            end
        end
    end
    Nq = Nq .* V;
end
end
