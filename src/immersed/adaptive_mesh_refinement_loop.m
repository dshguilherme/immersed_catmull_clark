function [mesh, u_all, vm_stresses, history] = adaptive_mesh_refinement_loop(octree_init, brep, bc_fn, opts)
% ADAPTIVE_MESH_REFINEMENT_LOOP
% Executes an automated mechanics-driven AMR cycle:
%   Solve -> Stress Recovery -> Error Indicator -> Mark -> Subdivide -> 2:1 Balance -> Re-solve
%
% Inputs:
%   octree_init - Initial octree (from octree_mesh_3d)
%   brep        - (Optional) B-Rep CAD model
%   bc_fn       - Function handle @(mesh) returning [fixed_dofs, F_master]
%   opts        - Struct with AMR control parameters:
%                   .max_amr_cycles - Number of AMR cycles (default: 3)
%                   .theta_dorfler  - Dorfler marking threshold (default: 0.35)
%                   .alpha_jump     - Stress jump indicator factor (default: 1.5)
%                   .max_level_limit- Max depth level (default: 4)
%                   .E, .nu         - Material properties
%
% Outputs:
%   mesh        - Final refined structural mesh
%   u_all       - Final displacement vector
%   vm_stresses - Final von Mises stress field
%   history     - Struct tracking convergence across AMR cycles

if nargin < 4, opts = struct(); end
if ~isfield(opts, 'max_amr_cycles'), opts.max_amr_cycles = 3; end
if ~isfield(opts, 'theta_dorfler'), opts.theta_dorfler = 0.35; end
if ~isfield(opts, 'alpha_jump'), opts.alpha_jump = 1.5; end
if ~isfield(opts, 'max_level_limit'), opts.max_level_limit = 4; end
if ~isfield(opts, 'E'), opts.E = 1.0e5; end
if ~isfield(opts, 'nu'), opts.nu = 0.3; end

current_octree = octree_init;
n_cycles = opts.max_amr_cycles;

history.dofs = zeros(n_cycles, 1);
history.n_elements = zeros(n_cycles, 1);
history.compliance = zeros(n_cycles, 1);
history.sigma_max = zeros(n_cycles, 1);
history.cond_K = zeros(n_cycles, 1);
history.meshes = cell(n_cycles, 1);
history.stresses = cell(n_cycles, 1);
history.displacements = cell(n_cycles, 1);

fprintf('========================================================================\n');
fprintf('  STARTING 3D ADAPTIVE MESH REFINEMENT (AMR) LOOP (%d CYCLES)\n', n_cycles);
fprintf('========================================================================\n');

for cycle = 1:n_cycles
    fprintf('\n--- AMR Cycle %d / %d ---\n', cycle, n_cycles);
    t_start = tic;
    
    % 1. Build structural mesh on current octree
    mesh = octree_structural_mesh(current_octree, brep, opts);
    
    % 2. Setup boundary conditions
    [fixed_dofs, F_master] = bc_fn(mesh);
    free_dofs = setdiff(1:3*mesh.n_master, fixed_dofs);
    
    % 3. Solve structural elasticity
    t_sol = tic;
    u_master = zeros(3*mesh.n_master, 1);
    u_master(free_dofs) = mesh.K_master(free_dofs, free_dofs) \ F_master(free_dofs);
    u_all = mesh.T_3d * u_master;
    t_solve = toc(t_sol);
    
    % 4. Compliance and Condition number
    compliance = F_master' * u_master;
    cond_K = condest(mesh.K_master(free_dofs, free_dofs));
    
    % 5. Stress recovery & AMR indicator
    [indicators, vm_stresses, marked_elems] = compute_amr_stress_indicators(mesh, u_master, opts);
    sigma_max = max(vm_stresses);
    
    % Record cycle history
    history.dofs(cycle) = numel(free_dofs);
    history.n_elements(cycle) = mesh.n_elements;
    history.compliance(cycle) = compliance;
    history.sigma_max(cycle) = sigma_max;
    history.cond_K(cycle) = cond_K;
    history.meshes{cycle} = mesh;
    history.stresses{cycle} = vm_stresses;
    history.displacements{cycle} = u_all;
    
    fprintf('Cycle %d Summary [Elapsed: %.2f s, Solve: %.2f s]:\n', cycle, toc(t_start), t_solve);
    fprintf('  Free DOFs: %d | Elements: %d | Peak vM Stress: %.4e | Compliance: %.4e | Cond(K): %.2e\n', ...
        numel(free_dofs), mesh.n_elements, sigma_max, compliance, cond_K);
    
    % 6. Refine for next cycle if not at final step
    if cycle < n_cycles
        if isempty(marked_elems)
            fprintf('No elements marked for refinement. Stopping AMR loop early.\n');
            n_cycles = cycle;
            break;
        end
        fprintf('Subdividing %d marked elements and enforcing 2:1 balancing...\n', numel(marked_elems));
        current_octree = subdivide_octree_leaves(current_octree, marked_elems);
    end
end

% Trim history arrays if terminated early
history.dofs = history.dofs(1:n_cycles);
history.n_elements = history.n_elements(1:n_cycles);
history.compliance = history.compliance(1:n_cycles);
history.sigma_max = history.sigma_max(1:n_cycles);
history.cond_K = history.cond_K(1:n_cycles);
history.meshes = history.meshes(1:n_cycles);
history.stresses = history.stresses(1:n_cycles);
history.displacements = history.displacements(1:n_cycles);

fprintf('\n========================================================================\n');
fprintf('  AMR LOOP COMPLETED: Final DOFs = %d, Peak Stress = %.4e\n', ...
    history.dofs(end), history.sigma_max(end));
fprintf('========================================================================\n\n');

end
