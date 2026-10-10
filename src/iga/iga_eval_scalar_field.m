function val = iga_eval_scalar_field(sp, u_sc, what)
% IGA_EVAL_SCALAR_FIELD  Evaluate a scalar spline field at the Gauss points.
%   val = iga_eval_scalar_field(sp, u_sc, 'value')     -> [nqn x nel]
%   val = iga_eval_scalar_field(sp, u_sc, 'gradient')  -> [dim x nqn x nel] (physical)
%   u_sc holds the sp.ndof_sc control values of one scalar component.

ue = reshape(u_sc(sp.conn_sc), 1, sp.nsh_sc, sp.nel);
switch lower(what)
    case 'value'
        Nq = iga_shape_functions(sp);
        val = reshape(sum(Nq .* ue, 2), sp.nqn, sp.nel);
    case 'gradient'
        [~, dNq] = iga_shape_functions(sp);
        val = reshape(sum(dNq .* reshape(ue, 1, 1, sp.nsh_sc, sp.nel), 3), sp.dim, sp.nqn, sp.nel);
    otherwise
        error('iga_eval_scalar_field: unknown option ''%s''.', what);
end
end
