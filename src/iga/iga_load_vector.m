function F = iga_load_vector(sp, f)
% IGA_LOAD_VECTOR  Consistent body-force vector F_(A,c) = int f_c phi_A.
%   F = iga_load_vector(sp, f)
%     f  handle f(x1, x2[, x3]) taking [nqn x nel] coordinate arrays and returning
%        [dim x nqn x nel] force values
%   Returns a column vector of length sp.ndof (component-blocked).

[X, W] = iga_quad_points(sp);
fq = f(X{:});
assert(isequal(size(fq), [sp.dim, sp.nqn, sp.nel]), 'iga_load_vector: f must return [dim x nqn x nel].');
Nq = iga_shape_functions(sp);
F = zeros(sp.ndof, 1);
for c = 1:sp.dim
    wf = reshape(fq(c, :, :), sp.nqn, 1, sp.nel) .* reshape(W, sp.nqn, 1, sp.nel);
    Fe = reshape(sum(Nq .* wf, 1), sp.nsh_sc, sp.nel);
    F = F + accumarray(sp.conn_sc(:) + (c - 1) * sp.ndof_sc, Fe(:), [sp.ndof, 1]);
end
end
