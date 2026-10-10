% test_gpu_approaches.m

L = 1; h = 0.5;
problem_data = cantilever_beam(L, h);
method_data.degree = [3 3];
method_data.regularity = [2 2];
method_data.nsub = [40 20];
method_data.nquad = [4 4];

[geometry, msh, sp] = buildSpaces(problem_data, method_data);
xPhys = ones(method_data.nsub);

fprintf('Testing CPU Elasticity_WQ...\n');
t0 = tic;
K_cpu = Elasticity_WQ(msh, sp, geometry, 1, 0.3, xPhys);
t_cpu = toc(t0);
fprintf('CPU time: %.4f s\n', t_cpu);

% Test GeoPDEs reference
fprintf('Testing GeoPDEs op_su_ev_tp...\n');
t1 = tic;
K_tp = op_su_ev_tp(sp, sp, msh, problem_data.lambda_lame, problem_data.mu_lame);
t_tp = toc(t1);
fprintf('GeoPDEs time: %.4f s\n', t_tp);

diff_norm = norm(K_cpu - K_tp, 'fro') / norm(K_tp, 'fro');
fprintf('Relative Frobenius difference: %.2e\n', diff_norm);
