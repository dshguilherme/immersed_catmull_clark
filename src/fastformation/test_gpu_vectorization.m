% test_gpu_vectorization.m
addpath(genpath('C:\Users\dshgu\OneDrive\Documents\geopdes-master'));

L = 1; h = 0.5;
problem_data = cantilever_beam(L, h);
method_data.degree = [4 4];
method_data.regularity = [3 3];
method_data.nsub = [40 20];
method_data.nquad = [5 5];

[geometry, msh, sp] = buildSpaces(problem_data, method_data);

% Precompute 1D rules and structures
for idim = 1:msh.ndim
    sp1d = sp.scalar_spaces{1}.sp_univ(idim);
    Connectivity(idim).neighbors = cellfun (@(x) unique (sp1d.connectivity(:,x)).', sp1d.supp, 'UniformOutput', false);
    Connectivity(idim).num_neigh = cellfun (@numel, Connectivity(idim).neighbors);
    Quad_rules(idim) = quadrule_stiff_fast(sp1d);
    brk{idim} = [sp.scalar_spaces{1}.knots{idim}(1), sp.scalar_spaces{1}.knots{idim}(end)];
    qn{idim} = Quad_rules(idim).all_points';
end

new_msh = msh_cartesian (brk, qn, [], geometry);
space_wq = sp.constructor (new_msh);

for idim = 1:msh.ndim
    sp1d = space_wq.scalar_spaces{1}.sp_univ(idim);
    for ii = 1:sp1d.ndof
        BSval{idim,ii} = sp1d.shape_functions(Quad_rules(idim).ind_points{ii}, Connectivity(idim).neighbors{ii}).'; 
        BSder{idim,ii} = sp1d.shape_function_gradients(Quad_rules(idim).ind_points{ii}, Connectivity(idim).neighbors{ii}).'; 
    end 	
end

nsd = msh.ndim;
aux_size = cellfun (@numel, qn);
jac = msh.map_der(qn);
E = zeros(3,2,2);
E(:,:,1) = [1 0; 0 0; 0 1];
E(:,:,2) = [0 0; 0 1; 1 0];
Y = 1; v = 0.3;
lambda = Y*v/(1+v)/(1-2*v);
mu = Y/2/(1+v);
D1 = lambda*ones(nsd) + 2*mu*eye(nsd);
D2 = mu*eye(1);
D = blkdiag(D1, D2);

C = C_ijkl(E, jac, D, nsd, aux_size);

fprintf('Running CPU Stiff_fast...\n');
t0 = tic;
K11_cpu = Stiff_fast(sp, Quad_rules, Connectivity, BSval, BSder, C{1,1});
t_cpu = toc(t0);
fprintf('CPU Stiff_fast time: %.4f s\n', t_cpu);
