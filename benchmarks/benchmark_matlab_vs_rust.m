% BENCHMARK_MATLAB_VS_RUST
% Runs identical benchmarks in MATLAB for direct comparison against Rust.

this_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(this_dir, '..');
addpath(genpath(fullfile(root_dir, 'src')));
addpath(fullfile(root_dir, 'tests'));

fprintf('========================================================================\n');
fprintf('  MATLAB BENCHMARK SUITE (CPU & GPU)\n');
fprintf('========================================================================\n');

has_gpu = false;
try
    d = gpuDevice();
    has_gpu = true;
    fprintf('--> Active MATLAB GPU: %s\n', d.Name);
catch
    fprintf('--> No GPU available for MATLAB gpuArray\n');
end

%% 1. Matrix-Free MatVec Benchmark (500, 2000, 8000, 32000 elements)
elem_counts = [500, 2000, 8000, 32000];
matvec_cpu_times = zeros(size(elem_counts));
matvec_gpu_times = zeros(size(elem_counts));

Ke = randn(24, 24, 'single');
Ke = Ke' * Ke; % symmetric positive semi-definite

for idx = 1:length(elem_counts)
    Ne = elem_counts(idx);
    pe = randn(24, Ne, 'single');
    we = rand(Ne, 1, 'single');
    
    % CPU timing (average over 50 runs)
    for warm = 1:5
        ye_cpu = Ke * pe .* reshape(we, 1, Ne);
    end
    tic;
    for rep = 1:50
        ye_cpu = Ke * pe .* reshape(we, 1, Ne);
    end
    matvec_cpu_times(idx) = (toc / 50) * 1e3; % in ms
    
    if has_gpu
        Ke_gpu = gpuArray(Ke);
        pe_gpu = gpuArray(pe);
        we_gpu = gpuArray(reshape(we, 1, Ne));
        % Warmup
        for warm = 1:10
            ye_gpu = (Ke_gpu * pe_gpu) .* we_gpu;
            wait(d);
        end
        tic;
        for rep = 1:50
            ye_gpu = (Ke_gpu * pe_gpu) .* we_gpu;
            wait(d);
        end
        matvec_gpu_times(idx) = (toc / 50) * 1e3; % in ms
    end
    
    fprintf('  MatVec Ne=%-5d | Total DOFs: %-6d | MATLAB CPU: %6.3f ms | MATLAB GPU: %6.3f ms\n', ...
        Ne, Ne * 24, matvec_cpu_times(idx), matvec_gpu_times(idx));
end

%% 2. Cox-de Boor 1D B-spline Basis Evaluation
p = 3;
knots = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1];
n_pts = 1000000;
u_pts = linspace(0, 1, n_pts);

% Warmup
evaluate_bspline_basis_1d(p, knots, 0.5);
tic;
for rep = 1:5
    B = evaluate_bspline_basis_1d(p, knots, u_pts);
end
bspline_time_ms = (toc / 5) * 1e3;
fprintf('\n--> Cox-de Boor (1M points, p=3): MATLAB Time = %6.2f ms\n', bspline_time_ms);

%% 3. Octree 2:1 Balancing & Structural Mesh
tic;
oct_p = octree_mesh_3d([0 2; 0 2; 0 2], [2 2 2], 1);
oct_p = subdivide_octree_leaves(oct_p, [1 4]);
balanced_octree = balance_octree_3d(oct_p);
struct_mesh = octree_structural_mesh(balanced_octree);
octree_time_ms = toc * 1e3;
fprintf('--> Octree 2:1 Balancing + MPC Assembly: MATLAB Time = %6.2f ms\n', octree_time_ms);

%% 4. Full Obstacle Course (5 Obstacles)
tic;
res = run_complete_obstacle_course();
obstacle_time_ms = toc * 1e3;
fprintf('--> Obstacle Course (5 Obstacles Total): MATLAB Time = %6.2f ms\n', obstacle_time_ms);

fprintf('========================================================================\n');
fprintf('  MATLAB BENCHMARK COMPLETE\n');
fprintf('========================================================================\n');
