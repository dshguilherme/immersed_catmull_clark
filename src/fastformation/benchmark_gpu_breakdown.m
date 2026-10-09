% benchmark_gpu_breakdown.m
% h-refinement timing with phase separation (p = 3, random element density).
%   - setup (geometry-only, amortized over the optimization) reported separately
%   - per-iteration formation: kernel / extract / transfer / sparse, device-synchronized
%   - median of nrep runs after one warm-up run
clearvars; clc;
addpath(genpath('C:\Users\dshgu\OneDrive\Documents\geopdes-master'));
addpath('c:\Users\dshgu\OneDrive\Documents\FastFormation');

p = 3; nrep = 5; penal = 3; Emin = 1e-3;
nsub_list = {[20,10],[40,20],[80,40],[120,60],[160,80],[200,100],[260,130],[320,160]};
nm = numel(nsub_list);
g = gpuDevice; reset(g);

f = {'dofs','t_ref','setup_cpu','setup_gpu','cpu_kernel','cpu_extract','cpu_sparse', ...
     'gpu64_kernel','gpu64_extract','gpu64_up','gpu64_down','gpu64_sparse', ...
     'gpu32_kernel','gpu32_extract','gpu32_down'};
for k = 1:numel(f), R.(f{k}) = nan(1, nm); end

rng(42);
fprintf('%-10s %7s | %8s | %8s %8s | %8s %8s %8s | %8s\n', 'mesh', 'DOFs', 'RefLoop', ...
    'CPUform', 'CPUkern', 'GPU64krn', 'GPU64e2e', 'GPU32krn', 'xfer64');
for m = 1:nm
    nsub = nsub_list{m};
    pd = cantilever_beam(1.0, 0.5);
    md.degree = [p p]; md.regularity = [p-1 p-1]; md.nsub = nsub; md.nquad = [p+1 p+1];
    [geometry, msh, sp] = buildSpaces(pd, md);
    R.dofs(m) = sp.ndof;
    x = rand(nsub);

    % Reference row-loop CPU implementation (the one used in the paper so far)
    if sp.ndof <= 70000
        tic; fast_stiffness_assembly(msh, sp, geometry, 1.0, 0.3, x, 'element', penal, Emin, true);
        R.t_ref(m) = toc;
    end

    % ---- Batched CPU ----
    Sc = wq_setup(msh, sp, geometry, 1.0, 0.3, 'cpu', 'double'); R.setup_cpu(m) = Sc.t_setup;
    wq_form(Sc, x, 'element', penal, Emin, true, 'cpu_sparse');
    T = zeros(nrep, 3);
    for r = 1:nrep
        [~, t] = wq_form(Sc, x, 'element', penal, Emin, true, 'cpu_sparse');
        T(r, :) = [t.kernel, t.extract, t.sparse];
    end
    T = median(T, 1); R.cpu_kernel(m) = T(1); R.cpu_extract(m) = T(2); R.cpu_sparse(m) = T(3);
    clear Sc

    % ---- GPU FP64 ----
    try
        Sg = wq_setup(msh, sp, geometry, 1.0, 0.3, 'gpu', 'double'); R.setup_gpu(m) = Sg.t_setup;
        wq_form(Sg, x, 'element', penal, Emin, true, 'cpu_sparse');
        T = zeros(nrep, 5);
        for r = 1:nrep
            [~, t] = wq_form(Sg, x, 'element', penal, Emin, true, 'cpu_sparse');
            T(r, :) = [t.kernel, t.extract, t.upload, t.download, t.sparse];
        end
        T = median(T, 1);
        R.gpu64_kernel(m) = T(1); R.gpu64_extract(m) = T(2); R.gpu64_up(m) = T(3);
        R.gpu64_down(m) = T(4); R.gpu64_sparse(m) = T(5);
        clear Sg
    catch ME
        fprintf('  GPU FP64 failed at %d DOFs: %s\n', sp.ndof, ME.message); clear Sg
    end
    reset(g);

    % ---- GPU FP32 ----
    try
        Sg = wq_setup(msh, sp, geometry, 1.0, 0.3, 'gpu', 'single');
        wq_form(Sg, x, 'element', penal, Emin, true, 'values');
        T = zeros(nrep, 3);
        for r = 1:nrep
            [~, t] = wq_form(Sg, x, 'element', penal, Emin, true, 'values');
            T(r, :) = [t.kernel, t.extract, t.download];
        end
        T = median(T, 1);
        R.gpu32_kernel(m) = T(1); R.gpu32_extract(m) = T(2); R.gpu32_down(m) = T(3);
        clear Sg
    catch ME
        fprintf('  GPU FP32 failed at %d DOFs: %s\n', sp.ndof, ME.message); clear Sg
    end
    reset(g);

    cpu_form = R.cpu_kernel(m) + R.cpu_extract(m) + R.cpu_sparse(m);
    g64_e2e = R.gpu64_kernel(m) + R.gpu64_extract(m) + R.gpu64_up(m) + R.gpu64_down(m) + R.gpu64_sparse(m);
    fprintf('%4dx%-5d %7d | %8.4f | %8.4f %8.4f | %8.4f %8.4f %8.4f | %8.4f\n', nsub(1), nsub(2), ...
        R.dofs(m), R.t_ref(m), cpu_form, R.cpu_kernel(m), R.gpu64_kernel(m) + R.gpu64_extract(m), ...
        g64_e2e, R.gpu32_kernel(m) + R.gpu32_extract(m), R.gpu64_up(m) + R.gpu64_down(m));
end
save('benchmark_gpu_breakdown.mat', 'R', 'nsub_list', 'p');
