% test_wq_batched.m - correctness check of batched WQ kernel vs. reference row loop
addpath(genpath('C:\Users\dshgu\OneDrive\Documents\geopdes-master'));
addpath('c:\Users\dshgu\OneDrive\Documents\FastFormation');
rng(1);
p = 3; nsub = [20, 10];
problem_data = cantilever_beam(1.0, 0.5);
method_data.degree = [p p]; method_data.regularity = [p-1 p-1];
method_data.nsub = nsub; method_data.nquad = [p+1 p+1];
[geometry, msh, sp] = buildSpaces(problem_data, method_data);
ncp = sp.scalar_spaces{1}.ndof_dir;

cases = {'element', rand(nsub); 'spline', rand(ncp)};
for c = 1:2
    typ = cases{c, 1}; x = cases{c, 2};
    K_ref = fast_stiffness_assembly(msh, sp, geometry, 1.0, 0.3, x, typ, 3, 1e-3, true);
    for dev = {'cpu', 'gpu'}
        for prec = {'double', 'single'}
            S = wq_setup(msh, sp, geometry, 1.0, 0.3, dev{1}, prec{1});
            K = wq_form(S, x, typ, 3, 1e-3, true, 'cpu_sparse');
            err = norm(K - K_ref, 'fro') / norm(K_ref, 'fro');
            fprintf('%-8s %-4s %-7s  rel. Frobenius diff = %.3e\n', typ, dev{1}, prec{1}, err);
        end
    end
end
