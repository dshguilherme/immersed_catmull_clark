function [indicators, vm_stresses, marked_elems, stresses_all] = compute_amr_stress_indicators(mesh, u_in, opts)
% COMPUTE_AMR_STRESS_INDICATORS
% Computes element-level stress fields, stress gradients / jumps,
% and an automated mechanics-driven AMR refinement indicator for 3D octrees.
%
% Stress Concentration Indicator:
%   eta_e = sigma_vM,e * h_e^{1/2} * (1 + alpha_jump * |sigma_vM,e - sigma_nbr,e| / (sigma_nbr,e + eps))
%
% Inputs:
%   mesh       - Output struct from octree_structural_mesh
%   u_in       - Global displacement vector (either u_master or u_all)
%   opts       - Struct of AMR options:
%                  .alpha_jump      - Gradient jump weighting factor (default: 1.5)
%                  .theta_dorfler   - Dorfler marking threshold fraction (default: 0.35)
%                  .strategy        - 'dorfler' (default) or 'top_fraction'
%                  .top_fraction    - Element fraction to mark if using top_fraction (default: 0.25)
%                  .max_level_limit - Max allowed octree level (default: 4)
%
% Outputs:
%   indicators   - [N_elem x 1] Refinement indicators eta_e
%   vm_stresses  - [N_elem x 1] Centroid von Mises stresses
%   marked_elems - Indices of elements flagged for subdivision
%   stresses_all - [N_elem x 6] Full Cauchy stress components

if nargin < 3, opts = struct(); end
if ~isfield(opts, 'alpha_jump'), opts.alpha_jump = 1.5; end
if ~isfield(opts, 'theta_dorfler'), opts.theta_dorfler = 0.35; end
if ~isfield(opts, 'strategy'), opts.strategy = 'dorfler'; end
if ~isfield(opts, 'top_fraction'), opts.top_fraction = 0.25; end
if ~isfield(opts, 'max_level_limit'), opts.max_level_limit = 4; end

% Expand displacement if given on master DOFs
if numel(u_in) == 3 * mesh.n_master
    u_all = mesh.T_3d * u_in;
else
    u_all = u_in;
end

n_elem = mesh.n_elements;
C_mat = mesh.C_tensor;
elem_nodes = mesh.elem_nodes;
h_elem = mesh.h_elem;

xi_c  = [-1,  1,  1, -1, -1,  1,  1, -1];
eta_c = [-1, -1,  1,  1, -1, -1,  1,  1];
zt_c  = [-1, -1, -1, -1,  1,  1,  1,  1];

stresses_all = zeros(n_elem, 6);
vm_stresses  = zeros(n_elem, 1);

% Centroid stress evaluation for each active element
for e = 1:n_elem
    en = elem_nodes(e, :);
    he = h_elem(e, :);
    
    % Gather 24 element displacement DOFs
    edofs = [(en-1)*3+1; (en-1)*3+2; (en-1)*3+3];
    ue = u_all(edofs(:));
    
    % B matrix at reference centroid (0,0,0)
    dNdxi  = (2/he(1)) * 0.125 * xi_c;
    dNdeta = (2/he(2)) * 0.125 * eta_c;
    dNdzt  = (2/he(3)) * 0.125 * zt_c;
    
    B = zeros(6, 24);
    for a = 1:8
        col = (a-1)*3 + (1:3);
        B(:, col) = [dNdxi(a), 0, 0;
                     0, dNdeta(a), 0;
                     0, 0, dNdzt(a);
                     0, dNdzt(a), dNdeta(a);
                     dNdzt(a), 0, dNdxi(a);
                     dNdeta(a), dNdxi(a), 0];
    end
    
    eps_e = B * ue;
    sig_e = C_mat * eps_e;
    stresses_all(e, :) = sig_e';
    
    % Von Mises stress
    % sig_e = [s11, s22, s33, s23, s13, s12]
    s11 = sig_e(1); s22 = sig_e(2); s33 = sig_e(3);
    s23 = sig_e(4); s13 = sig_e(5); s12 = sig_e(6);
    
    vm_stresses(e) = sqrt(0.5 * ((s11 - s22)^2 + (s22 - s33)^2 + (s33 - s11)^2 + ...
                          6.0 * (s23^2 + s13^2 + s12^2)));
end

% Element-node adjacency to compute neighborhood average stress
N_adj = sparse(elem_nodes', repmat(1:n_elem, 8, 1), 1, mesh.n_nodes, n_elem);
E_adj = (N_adj' * N_adj) > 0;
E_adj(1:n_elem+1:end) = 0; % Remove self-connection

deg = full(sum(E_adj, 2));
deg(deg == 0) = 1;
sigma_nbr = full(E_adj * vm_stresses) ./ deg;

% Stress jump across cell boundaries
stress_jump = abs(vm_stresses - sigma_nbr);
eps_reg = 1e-4 * max(vm_stresses) + 1e-8;

% Characteristic cell diameter h_e = (V_e)^(1/3)
h_char = (prod(h_elem, 2)).^(1/3);

% Automated Stress Concentration Indicator
indicators = vm_stresses .* sqrt(h_char) .* (1.0 + opts.alpha_jump * (stress_jump ./ (sigma_nbr + eps_reg)));

% Element Marking Strategy
if strcmp(opts.strategy, 'dorfler')
    [sorted_ind, sort_idx] = sort(indicators, 'descend');
    cum_energy = cumsum(sorted_ind.^2);
    total_energy = cum_energy(end);
    target_energy = opts.theta_dorfler * total_energy;
    n_mark = find(cum_energy >= target_energy, 1);
    if isempty(n_mark), n_mark = 1; end
    marked_candidates = sort_idx(1:n_mark);
else % 'top_fraction'
    [~, sort_idx] = sort(indicators, 'descend');
    n_mark = max(1, round(opts.top_fraction * n_elem));
    marked_candidates = sort_idx(1:n_mark);
end

% Enforce maximum subdivision level limit
marked_elems = marked_candidates(mesh.levels(marked_candidates) < opts.max_level_limit);

fprintf('AMR Stress Evaluation: Peak vM = %.3e, Mean vM = %.3e | Marked %d / %d elements for refinement\n', ...
    max(vm_stresses), mean(vm_stresses), numel(marked_elems), n_elem);

end
