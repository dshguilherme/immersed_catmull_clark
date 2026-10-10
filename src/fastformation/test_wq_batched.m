% test_wq_batched.m - correctness check of batched WQ kernel vs. reference row loop
rng(1);
p = 3; nsub = [20, 10];
sp = iga_space_box([0 1.0; 0 0.5], nsub, p);
ncp = sp.ndof_dir;

cases = {'element', rand(nsub); 'spline', rand(ncp)};
for c = 1:2
    typ = cases{c, 1}; x = cases{c, 2};
    K_ref = fast_stiffness_assembly(sp, 1.0, 0.3, x, typ, 3, 1e-3, true);
    for dev = {'cpu', 'gpu'}
        for prec = {'double', 'single'}
            S = wq_setup(sp, 1.0, 0.3, dev{1}, prec{1});
            K = wq_form(S, x, typ, 3, 1e-3, true, 'cpu_sparse');
            err = norm(K - K_ref, 'fro') / norm(K_ref, 'fro');
            fprintf('%-8s %-4s %-7s  rel. Frobenius diff = %.3e\n', typ, dev{1}, prec{1}, err);
        end
    end
end
