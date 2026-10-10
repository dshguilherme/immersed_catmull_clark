function r = iga_integrate_scalar(sp, Sq)
% IGA_INTEGRATE_SCALAR  Project quadrature-point data onto the scalar basis.
%   r = iga_integrate_scalar(sp, Sq) returns r_A = int phi_A S for S given at
%   the Gauss points as Sq [nqn x nel]. r has length sp.ndof_sc.

[~, W] = iga_quad_points(sp);
Nq = iga_shape_functions(sp);
re = reshape(sum(Nq .* reshape(W .* Sq, sp.nqn, 1, sp.nel), 1), sp.nsh_sc, sp.nel);
r = accumarray(sp.conn_sc(:), re(:), [sp.ndof_sc, 1]);
end
