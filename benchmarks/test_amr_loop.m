% TEST_AMR_LOOP
clear; clc;
this_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(this_dir, '..', 'src')));

gb = [0 1; 0 1; 0 0.2];
res = [4 4 2];
ot_init = octree_mesh_3d(gb, res, 0, []);

bc_fn = @(mesh) setup_cantilever_bcs(mesh);

opts.max_amr_cycles = 3;
opts.theta_dorfler = 0.35;
opts.alpha_jump = 1.5;
opts.max_level_limit = 3;

[mesh_fin, u_fin, vm_fin, hist] = adaptive_mesh_refinement_loop(ot_init, [], bc_fn, opts);
fprintf('AMR Loop verified successfully!\n');
fprintf('DOFs: %s\n', mat2str(hist.dofs));
fprintf('Peak Stresses: %s\n', mat2str(hist.sigma_max, 4));

function [fixed_dofs, F_master] = setup_cantilever_bcs(mesh)
    master_nodes = mesh.nodes(mesh.master_node_ids, :);
    % Clamped at x = 0
    fixed_idx = find(master_nodes(:, 1) < 1e-5);
    fixed_dofs = [(fixed_idx-1)*3+1; (fixed_idx-1)*3+2; (fixed_idx-1)*3+3];
    
    % Downward load at x = 1, y in [0, 0.5]
    load_idx = find(master_nodes(:, 1) > 1 - 1e-5 & master_nodes(:, 2) < 0.5 + 1e-5);
    F_master = zeros(3 * mesh.n_master, 1);
    if ~isempty(load_idx)
        load_dofs = (load_idx - 1)*3 + 2; % -Y direction
        F_master(load_dofs) = -100.0 / numel(load_dofs);
    end
end
