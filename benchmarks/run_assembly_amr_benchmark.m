% RUN_ASSEMBLY_AMR_BENCHMARK
% End-to-end multi-body CAD assembly structural analysis with:
% 1. Independent non-conforming background octrees for each body
% 2. Weak bonded contact coupling via Nitsche's formulation
% 3. Automated mechanics-driven adaptive mesh refinement (AMR) with 2:1 balancing
% 4. Convergence of peak von Mises stresses and compliance across AMR cycles
clear; clc; close all;

this_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(this_dir, '..', 'src')));

fprintf('========================================================================\n');
fprintf('  MULTI-BODY CAD ASSEMBLY IMMERSED AMR BENCHMARK\n');
fprintf('========================================================================\n\n');

%% 1. Load Body A (NIST CTC-01 STEP Model)
stp_path = fullfile(this_dir, '..', 'Models', 'NIST-PMI-STEP-Files', 'NIST-PMI-STEP-Files', ...
                    'AP203 geometry only', 'nist_ctc_01_asme1_rd.stp');
brepA = importBRep(stp_path);

minA = min(brepA.nodes); maxA = max(brepA.nodes);
fprintf('Body A (NIST CTC-01): [%.1f, %.1f] x [%.1f, %.1f] x [%.1f, %.1f]\n', ...
    minA(1), maxA(1), minA(2), maxA(2), minA(3), maxA(3));

%% 2. Create Mating Body B (Punch / Flange Block)
% Body B rests directly on the main deck of Body A at Z = 0
% with dimensions [80, 350] x [-150, 150] x [0, 80]
bx1 = 80;  bx2 = 350;
by1 = -150; by2 = 150;
bz1 = 0;   bz2 = 80;

box_v = [bx1 by1 bz1; bx2 by1 bz1; bx2 by2 bz1; bx1 by2 bz1; ...
         bx1 by1 bz2; bx2 by1 bz2; bx2 by2 bz2; bx1 by2 bz2];
box_f = [1 2 3; 1 3 4; ... % -Z bottom (contact face)
         5 7 6; 5 8 7; ... % +Z top
         1 6 2; 1 5 6; ... % -Y
         4 3 7; 4 7 8; ... % +Y
         1 4 8; 1 8 5; ... % -X
         2 7 3; 2 6 7];    % +X

% Subdivide box surface for richer contact representation
n_sub_b = 2;
b_nodes = box_v;
b_elem = box_f;
for s = 1:n_sub_b
    n_tri = size(b_elem, 1);
    v1 = b_nodes(b_elem(:,1), :);
    v2 = b_nodes(b_elem(:,2), :);
    v3 = b_nodes(b_elem(:,3), :);
    m12 = 0.5 * (v1 + v2);
    m23 = 0.5 * (v2 + v3);
    m31 = 0.5 * (v3 + v1);
    
    nv0 = size(b_nodes, 1);
    new_v = [b_nodes; m12; m23; m31];
    [u_nodes, ~, map] = unique(round(new_v, 4), 'rows', 'stable');
    
    id1 = map(b_elem(:,1));
    id2 = map(b_elem(:,2));
    id3 = map(b_elem(:,3));
    id12 = map(nv0 + (1:n_tri)');
    id23 = map(nv0 + n_tri + (1:n_tri)');
    id31 = map(nv0 + 2*n_tri + (1:n_tri)');
    
    new_elem = [id1, id12, id31; ...
                id12, id2, id23; ...
                id31, id23, id3; ...
                id12, id23, id31];
    b_nodes = u_nodes;
    b_elem = new_elem;
end
brepB.nodes = b_nodes;
brepB.elements = b_elem;

fprintf('Body B (Mating Block): %d nodes, %d facets\n', size(brepB.nodes, 1), size(brepB.elements, 1));

%% 3. Initialize Independent Octree Grids
padA = 0.05 * (maxA - minA);
gbA = [minA(1)-padA(1), maxA(1)+padA(1); ...
       minA(2)-padA(2), maxA(2)+padA(2); ...
       minA(3)-padA(3), maxA(3)+padA(3)];
resA = [8, 5, 3]; % Body A initial resolution

padB = [20, 20, 10];
gbB = [bx1-padB(1), bx2+padB(1); ...
       by1-padB(2), by2+padB(2); ...
       bz1-padB(3), bz2+padB(3)];
resB = [6, 4, 3]; % Body B initial resolution (non-conforming)

octreeA = octree_mesh_3d(gbA, resA, 0, []);
octreeB = octree_mesh_3d(gbB, resB, 0, []);

%% 4. Multi-Body Assembly AMR Loop
n_amr_cycles = 3;
history.dofs = zeros(n_amr_cycles, 1);
history.elems = zeros(n_amr_cycles, 1);
history.compliance = zeros(n_amr_cycles, 1);
history.peak_stress = zeros(n_amr_cycles, 1);
history.cond_K = zeros(n_amr_cycles, 1);
history.meshA = cell(n_amr_cycles, 1);
history.meshB = cell(n_amr_cycles, 1);
history.vmA = cell(n_amr_cycles, 1);
history.vmB = cell(n_amr_cycles, 1);
history.uA = cell(n_amr_cycles, 1);
history.uB = cell(n_amr_cycles, 1);

mat_opts.E = 2.0e5; % Steel: 200 GPa
mat_opts.nu = 0.3;
contact_opts.gap_tol = 6.0;
contact_opts.gamma_c = 150.0;

for cycle = 1:n_amr_cycles
    fprintf('\n========================================================================\n');
    fprintf('  ASSEMBLY AMR CYCLE %d / %d\n', cycle, n_amr_cycles);
    fprintf('========================================================================\n');
    t_cyc = tic;
    
    % Step A: Build structural meshes for both bodies
    meshA = octree_structural_mesh(octreeA, brepA, mat_opts);
    meshB = octree_structural_mesh(octreeB, brepB, mat_opts);
    
    % Step B: Assemble Nitsche bonded contact coupling
    [K_contact, contact_info] = assemble_octree_nitsche_contact(meshA, meshB, brepA, brepB, contact_opts);
    
    % Step C: Coupled Assembly Stiffness
    K_assembly = blkdiag(meshA.K_master, meshB.K_master) + K_contact;
    ndof_tot = size(K_assembly, 1);
    
    % Step D: Boundary Conditions
    % Body A clamped at bottom face (Z < -90)
    masterA_nodes = meshA.nodes(meshA.master_node_ids, :);
    fixedA_idx = find(masterA_nodes(:, 3) < minA(3) + 15);
    fixedA_dofs = [(fixedA_idx-1)*3+1; (fixedA_idx-1)*3+2; (fixedA_idx-1)*3+3];
    
    % Body B loaded at top face (Z > bz2 - 5) with downward vertical pressure
    masterB_nodes = meshB.nodes(meshB.master_node_ids, :);
    loadB_idx = find(masterB_nodes(:, 3) > bz2 - 10);
    loadB_dofs_z = (3 * meshA.n_master) + (loadB_idx - 1)*3 + 3; % Z component in global block
    
    F_assembly = zeros(ndof_tot, 1);
    total_force = -2.0e4; % 20 kN downward
    F_assembly(loadB_dofs_z) = total_force / numel(loadB_dofs_z);
    
    free_dofs = setdiff(1:ndof_tot, fixedA_dofs);
    
    % Step E: Solve Assembly Elasticity
    t_sol = tic;
    u_assembly = zeros(ndof_tot, 1);
    u_assembly(free_dofs) = K_assembly(free_dofs, free_dofs) \ F_assembly(free_dofs);
    t_solve = toc(t_sol);
    
    % Unpack displacements
    u_mA = u_assembly(1:3*meshA.n_master);
    u_mB = u_assembly(3*meshA.n_master + (1:3*meshB.n_master));
    u_allA = meshA.T_3d * u_mA;
    u_allB = meshB.T_3d * u_mB;
    
    % Step F: Stress Recovery & Error Indicator targeting contact interface and geometric features
    amr_opts.alpha_jump = 1.8;
    amr_opts.theta_dorfler = 0.35;
    amr_opts.max_level_limit = 3;
    
    [indA, vmA, markA] = compute_amr_stress_indicators(meshA, u_mA, amr_opts);
    [indB, vmB, markB_raw] = compute_amr_stress_indicators(meshB, u_mB, amr_opts);
    % Focus Body B refinement near contact interface (Z <= 40)
    markB = markB_raw(meshB.bounds(markB_raw, 5) <= 40);
    if isempty(markB) && ~isempty(markB_raw), markB = markB_raw(1); end
    
    compliance = F_assembly' * u_assembly;
    peak_stress = max([max(vmA), max(vmB)]);
    cond_K = condest(K_assembly(free_dofs, free_dofs));
    
    % Record History
    history.dofs(cycle) = numel(free_dofs);
    history.elems(cycle) = meshA.n_elements + meshB.n_elements;
    history.compliance(cycle) = compliance;
    history.peak_stress(cycle) = peak_stress;
    history.cond_K(cycle) = cond_K;
    history.meshA{cycle} = meshA;
    history.meshB{cycle} = meshB;
    history.vmA{cycle} = vmA;
    history.vmB{cycle} = vmB;
    history.uA{cycle} = u_allA;
    history.uB{cycle} = u_allB;
    
    fprintf('AMR Cycle %d Complete [Elapsed: %.2f s, Solve: %.2f s]:\n', cycle, toc(t_cyc), t_solve);
    fprintf('  Total DOFs: %d | Total Elems: %d | Peak Stress: %.2f MPa | Compliance: %.4e | Cond(K): %.2e\n', ...
        ndof_tot, meshA.n_elements + meshB.n_elements, peak_stress, compliance, cond_K);
    
    % Step G: Refine octrees for next cycle
    if cycle < n_amr_cycles
        fprintf('Subdividing marked cells: %d in Body A, %d in Body B...\n', numel(markA), numel(markB));
        octreeA = subdivide_octree_leaves(octreeA, markA);
        octreeB = subdivide_octree_leaves(octreeB, markB);
    end
end

%% 5. Export Publication Quality Figure
fprintf('\nGenerating high-contrast publication figure...\n');
fig = figure('Color', 'w', 'Position', [40, 40, 1500, 900]);

% Panel A: Assembly Mesh and Contact Interface
subplot(2, 2, 1);
% Body A CAD surface
patch('Faces', brepA.elements, 'Vertices', brepA.nodes, ...
      'FaceColor', [0.85 0.88 0.92], 'EdgeColor', [0.6 0.65 0.75], 'EdgeAlpha', 0.3, 'FaceAlpha', 0.7);
hold on;
% Body B CAD surface
patch('Faces', brepB.elements, 'Vertices', brepB.nodes, ...
      'FaceColor', [0.95 0.82 0.75], 'EdgeColor', [0.8 0.55 0.45], 'EdgeAlpha', 0.3, 'FaceAlpha', 0.7);

% Plot refined octree leaf boundaries around contact zone
fin_mA = history.meshA{end};
fin_mB = history.meshB{end};
% Highlight fine elements (level >= 1)
fine_A = find(fin_mA.levels >= 1);
for fa = fine_A(:)'
    b = fin_mA.bounds(fa, :);
    plot3([b(1) b(2) b(2) b(1) b(1)], [b(3) b(3) b(4) b(4) b(3)], [b(5) b(5) b(5) b(5) b(5)], 'b-', 'LineWidth', 0.8);
    plot3([b(1) b(2) b(2) b(1) b(1)], [b(3) b(3) b(4) b(4) b(3)], [b(6) b(6) b(6) b(6) b(6)], 'b-', 'LineWidth', 0.8);
end
fine_B = find(fin_mB.levels >= 1);
for fb = fine_B(:)'
    b = fin_mB.bounds(fb, :);
    plot3([b(1) b(2) b(2) b(1) b(1)], [b(3) b(3) b(4) b(4) b(3)], [b(5) b(5) b(5) b(5) b(5)], 'r-', 'LineWidth', 0.8);
    plot3([b(1) b(2) b(2) b(1) b(1)], [b(3) b(3) b(4) b(4) b(3)], [b(6) b(6) b(6) b(6) b(6)], 'r-', 'LineWidth', 0.8);
end

% Contact interface points
if contact_info.n_pairs > 0
    scatter3(contact_info.points(:,1), contact_info.points(:,2), contact_info.points(:,3), ...
             25, [0.1 0.75 0.2], 'filled');
end

axis equal tight; grid on; box on; view(38, 26);
camlight('headlight'); camlight('right'); lighting gouraud;
set(gca, 'Color', 'w', 'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2], 'ZColor', [0.2 0.2 0.2]);
xlabel('X [mm]', 'FontWeight', 'bold', 'Color', 'k');
ylabel('Y [mm]', 'FontWeight', 'bold', 'Color', 'k');
zlabel('Z [mm]', 'FontWeight', 'bold', 'Color', 'k');
title('(a) Multi-Body Assembly & Non-Conforming Octree AMR', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

% Panel B: Von Mises Stress Distribution on Assembly
subplot(2, 2, 2);
vmA_end = history.vmA{end};
vmB_end = history.vmB{end};
% Map element stresses to centroids for scatter representation
centsA = 0.5 * (fin_mA.bounds(:, [1 3 5]) + fin_mA.bounds(:, [2 4 6]));
centsB = 0.5 * (fin_mB.bounds(:, [1 3 5]) + fin_mB.bounds(:, [2 4 6]));

scatter3(centsA(:,1), centsA(:,2), centsA(:,3), 45, vmA_end, 'filled');
hold on;
scatter3(centsB(:,1), centsB(:,2), centsB(:,3), 45, vmB_end, 'filled');
colormap(gca, turbo);
clim([0, prctile([vmA_end; vmB_end], 95)]);
cb = colorbar;
ylabel(cb, 'von Mises Stress \sigma_{vM} [MPa]', 'FontWeight', 'bold', 'FontSize', 10);
axis equal tight; grid on; box on; view(38, 26);
set(gca, 'Color', 'w', 'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2], 'ZColor', [0.2 0.2 0.2]);
xlabel('X [mm]', 'FontWeight', 'bold', 'Color', 'k');
ylabel('Y [mm]', 'FontWeight', 'bold', 'Color', 'k');
zlabel('Z [mm]', 'FontWeight', 'bold', 'Color', 'k');
title('(b) Von Mises Stress Field Under Contact Loading', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

% Panel C: Convergence of Peak Stress & Compliance
subplot(2, 2, 3);
yyaxis left;
plot(history.dofs, history.peak_stress, 'r-s', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
ylabel('Peak von Mises Stress \sigma_{max} [MPa]', 'FontWeight', 'bold', 'Color', [0.8 0 0]);
set(gca, 'YColor', [0.8 0 0]);

yyaxis right;
plot(history.dofs, history.compliance, 'b--o', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
ylabel('Strain Energy / Compliance C [N\cdot mm]', 'FontWeight', 'bold', 'Color', [0 0.2 0.8]);
set(gca, 'YColor', [0 0.2 0.8]);

grid on; box on;
set(gca, 'Color', 'w', 'XColor', [0.2 0.2 0.2], 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.7);
xlabel('Active Assembly Degrees of Freedom (DOFs)', 'FontWeight', 'bold', 'Color', 'k');
title('(c) AMR Convergence of Stress & Compliance', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

% Panel D: Condition Number Stability vs AMR Refinement
subplot(2, 2, 4);
semilogy(1:n_amr_cycles, history.cond_K, 'k-d', 'LineWidth', 2.2, 'MarkerSize', 8, 'MarkerFaceColor', [0.2 0.6 0.2]);
grid on; box on;
set(gca, 'Color', 'w', 'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2], 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.7);
xlabel('AMR Cycle Level', 'FontWeight', 'bold', 'Color', 'k');
ylabel('Matrix Condition Number \kappa(K)', 'FontWeight', 'bold', 'Color', 'k');
title('(d) Ghost Penalty Condition Stability across AMR Cycles', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xticks(1:n_amr_cycles);

fig_path = fullfile(this_dir, '..', 'figures', 'fig_octree_amr_assembly_mechanics.png');
exportgraphics(fig, fig_path, 'Resolution', 220);
close(fig);
fprintf('Exported assembly AMR benchmark figure: %s\n', fig_path);

% Save numerical workspace
save(fullfile(this_dir, '..', 'figures', 'assembly_amr_results.mat'), 'history', 'contact_info');
fprintf('Saved numerical benchmark workspace to figures/assembly_amr_results.mat\n');

fprintf('\n========================================================================\n');
fprintf('  ASSEMBLY AMR BENCHMARK COMPLETED SUCCESSFULLY\n');
fprintf('========================================================================\n');
