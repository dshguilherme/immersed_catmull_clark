function dC = fast_sensitivities(u, msh, space, geometry, xPhys, density_type, penal, Emin, YOUNG, POISSON, sp_rho)
% FAST_SENSITIVITIES Evaluates compliance sensitivities directly from the
% displacement solution without forming or storing element stiffness matrices.
%
% Inputs:
%   u            - Displacement state vector
%   msh          - GeoPDEs mesh structure
%   space        - GeoPDEs vector space
%   geometry     - GeoPDEs geometry
%   xPhys        - Physical density array
%   density_type - 'element' or 'spline'
%   penal        - SIMP penalization exponent (default: 3)
%   Emin         - Void modulus ratio (default: 1e-9)
%   YOUNG        - Solid Young's modulus (default: 1.0)
%   POISSON      - Poisson's ratio (default: 0.3)
%   sp_rho       - (Optional) B-spline space for density field
%
% Output:
%   dC           - Sensitivity gradient array (matching xPhys dimensions)

if nargin < 6 || isempty(density_type), density_type = 'element'; end
if nargin < 7 || isempty(penal), penal = 3; end
if nargin < 8 || isempty(Emin), Emin = 1e-9; end
if nargin < 9 || isempty(YOUNG), YOUNG = 1.0; end
if nargin < 10 || isempty(POISSON), POISSON = 0.3; end
if nargin < 11, sp_rho = []; end

nsd = msh.ndim;

% Compute solid constitutive matrix D0
lambda = YOUNG * POISSON / ((1 + POISSON) * (1 - 2 * POISSON));
mu = YOUNG / (2 * (1 + POISSON));
if nsd == 2
    % Plane strain (or plane stress)
    D0 = [lambda + 2*mu, lambda, 0; ...
          lambda, lambda + 2*mu, 0; ...
          0, 0, mu];
else
    D0 = [lambda + 2*mu, lambda, lambda, 0, 0, 0; ...
          lambda, lambda + 2*mu, lambda, 0, 0, 0; ...
          lambda, lambda, lambda + 2*mu, 0, 0, 0; ...
          0, 0, 0, mu, 0, 0; ...
          0, 0, 0, 0, mu, 0; ...
          0, 0, 0, 0, 0, mu];
end

if isa(space, 'sp_vector')
    sp_scalar = space.scalar_spaces{1};
else
    sp_scalar = space;
end

has_grads = false;
try
    if ~isempty(sp_scalar.shape_function_gradients)
        has_grads = true;
    end
catch
    has_grads = false;
end

if ~has_grads
    sp_scalar = sp_precompute(sp_scalar, msh, 'gradient', true);
end

u1 = u(1 : sp_scalar.ndof);
u2 = u(sp_scalar.ndof + 1 : 2 * sp_scalar.ndof);

grad_u1 = sp_eval_msh(u1, sp_scalar, msh, 'gradient'); % [2 x nqn x nel]
grad_u2 = sp_eval_msh(u2, sp_scalar, msh, 'gradient'); % [2 x nqn x nel]

% Extract engineering strains
if nsd == 2
    eps11 = squeeze(grad_u1(1, :, :));
    eps22 = squeeze(grad_u2(2, :, :));
    gamma12 = squeeze(grad_u1(2, :, :) + grad_u2(1, :, :));
    
    nqn = msh.nqn;
    nel = msh.nel;
    
    strain_energy_density = zeros(nqn, nel);
    for q = 1:nqn
        eps_q = [eps11(q, :); eps22(q, :); gamma12(q, :)]; % [3 x nel]
        D_eps = D0 * eps_q; % [3 x nel]
        strain_energy_density(q, :) = sum(eps_q .* D_eps, 1); % [1 x nel]
    end
end

% Quadrature weights and Jacobian determinants: [nqn x nel]
qw_J = msh.quad_weights .* msh.jacdet;

if strcmpi(density_type, 'element')
    % Option A: Element-wise sensitivity
    % Integrate unpenalized strain energy per element
    Ee = sum(qw_J .* strain_energy_density, 1); % [1 x nel]
    
    % Reshape Ee to match xPhys dimensions
    nel_dir = msh.nel_dir;
    Ee_grid = reshape(Ee, nel_dir);
    
    % Sensitivity: dC / drho_e = - penal * rho_e^(penal-1) * (1 - Emin) * Ee
    dC = - penal * (xPhys .^ (penal - 1)) * (1 - Emin) .* Ee_grid;
    
else
    % Option B: Continuous B-spline density field
    % Sensitivity density: S = penal * rho^(penal-1) * (1 - Emin) * (eps' * D0 * eps)
    if isempty(sp_rho)
        sp_rho = sp_scalar;
    end
    has_shp = false;
    try
        if ~isempty(sp_rho.shape_functions)
            has_shp = true;
        end
    catch
        has_shp = false;
    end
    if ~has_shp
        sp_rho = sp_precompute(sp_rho, msh, 'value', true);
    end
    
    % Evaluate rho at mesh quadrature points
    rho_at_q = sp_eval_msh(xPhys(:), sp_rho, msh, 'value');
    rho_at_q = squeeze(rho_at_q); % [nqn x nel]
    
    S_q = penal * (rho_at_q .^ (penal - 1)) * (1 - Emin) .* strain_energy_density; % [nqn x nel]
    
    % Project onto spline basis: \int R_K * S dOmega
    % op_f_v multiplies by quadrature weights and jacdet internally
    rhs_sens = op_f_v(sp_rho, msh, S_q);
    dC = - reshape(rhs_sens, size(xPhys));
end

end
