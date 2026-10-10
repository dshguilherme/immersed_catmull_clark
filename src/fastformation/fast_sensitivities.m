function dC = fast_sensitivities(u, space, xPhys, density_type, penal, Emin, YOUNG, POISSON, sp_rho)
% FAST_SENSITIVITIES Evaluates compliance sensitivities directly from the
% displacement solution without forming or storing element stiffness matrices.
%
% Inputs:
%   u            - Displacement state vector (component-blocked)
%   space        - displacement space on a box (see IGA_SPACE_BOX)
%   xPhys        - Physical density array
%   density_type - 'element' or 'spline'
%   penal        - SIMP penalization exponent (default: 3)
%   Emin         - Void modulus ratio (default: 1e-9)
%   YOUNG        - Solid Young's modulus (default: 1.0)
%   POISSON      - Poisson's ratio (default: 0.3)
%   sp_rho       - (Optional) space for the density field (default: space)
%
% Output:
%   dC           - Sensitivity gradient array (matching xPhys dimensions)

if nargin < 4 || isempty(density_type), density_type = 'element'; end
if nargin < 5 || isempty(penal), penal = 3; end
if nargin < 6 || isempty(Emin), Emin = 1e-9; end
if nargin < 7 || isempty(YOUNG), YOUNG = 1.0; end
if nargin < 8 || isempty(POISSON), POISSON = 0.3; end
if nargin < 9 || isempty(sp_rho), sp_rho = space; end

nsd = space.dim;
lambda = YOUNG * POISSON / ((1 + POISSON) * (1 - 2 * POISSON));
mu = YOUNG / (2 * (1 + POISSON));

% Displacement gradients at the Gauss points: grad_u{c} is [dim x nqn x nel]
ndof_sc = space.ndof_sc;
grad_u = cell(1, nsd);
for c = 1:nsd
    grad_u{c} = iga_eval_scalar_field(space, u((c - 1) * ndof_sc + (1:ndof_sc)), 'gradient');
end

% Unpenalized strain energy density eps : D0 : eps = lambda tr(eps)^2 + 2 mu eps:eps
tr_eps = zeros(space.nqn, space.nel);
eps_eps = zeros(space.nqn, space.nel);
for i = 1:nsd
    tr_eps = tr_eps + reshape(grad_u{i}(i, :, :), space.nqn, space.nel);
    for j = 1:nsd
        eij = 0.5 * (reshape(grad_u{i}(j, :, :), space.nqn, space.nel) + ...
                     reshape(grad_u{j}(i, :, :), space.nqn, space.nel));
        eps_eps = eps_eps + eij.^2;
    end
end
strain_energy_density = lambda * tr_eps.^2 + 2 * mu * eps_eps;

if strcmpi(density_type, 'element')
    % Option A: Element-wise sensitivity
    [~, W] = iga_quad_points(space);
    Ee = sum(W .* strain_energy_density, 1);
    Ee_grid = reshape(Ee, space.nel_dir);
    % dC / drho_e = - penal * rho_e^(penal-1) * (1 - Emin) * Ee
    dC = - penal * (xPhys .^ (penal - 1)) * (1 - Emin) .* Ee_grid;
else
    % Option B: Continuous B-spline density field on the displacement mesh
    rho_at_q = iga_eval_scalar_field(sp_rho, xPhys(:), 'value');
    S_q = penal * (rho_at_q .^ (penal - 1)) * (1 - Emin) .* strain_energy_density;
    % Project onto spline basis: int R_K * S dOmega
    rhs_sens = iga_integrate_scalar(sp_rho, S_q);
    dC = - reshape(rhs_sens, size(xPhys));
end
end
