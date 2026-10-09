classdef testGPUOctree < matlab.unittest.TestCase
    % TESTGPUOCTREE
    % Unit and integration tests for Phase 3: Matrix-Free GPU Octree Operator
    % Verifies exact matvec agreement with sparse K, and matrix-free PCG solver accuracy.
    
    properties
        mesh
        brep
    end
    
    methods(TestMethodSetup)
        function setupFixture(testCase)
            % Simple 3D box B-Rep [0, 4] x [0, 4] x [0, 4]
            nodes = [
                0, 0, 0; 4, 0, 0; 4, 4, 0; 0, 4, 0;
                0, 0, 4; 4, 0, 4; 4, 4, 4; 0, 4, 4
            ];
            elements = [
                1, 2, 6; 1, 6, 5;
                2, 3, 7; 2, 7, 6;
                3, 4, 8; 3, 8, 7;
                4, 1, 5; 4, 5, 8;
                1, 4, 3; 1, 3, 2;
                5, 6, 7; 5, 7, 8
            ];
            testCase.brep.nodes = nodes;
            testCase.brep.elements = elements;
            
            % Adaptive octree with 2 levels of refinement
            grid_bounds = [-1, 5; -1, 5; -1, 5];
            grid_res = [2, 2, 2];
            oct = octree_mesh_3d(grid_bounds, grid_res, 1);
            oct = subdivide_octree_leaves(oct, [1 3 5]);
            oct = balance_octree_3d(oct);
            
            opts.E = 2e5;
            opts.nu = 0.3;
            opts.fictitious_weight = 1e-3;
            testCase.mesh = octree_structural_mesh(oct, testCase.brep, opts);
        end
    end
    
    methods(Test)
        function testMatvecExactAgreement(testCase)
            % Compare matrix-free operator against explicit sparse K_master
            mesh = testCase.mesh;
            ndof = 3 * mesh.n_master;
            
            rng(42);
            p = randn(ndof, 1);
            
            % Sparse action
            y_sparse = mesh.K_master * p;
            
            % Matrix-free action
            y_matvec = gpu_octree_matvec(mesh, p, struct('use_gpu', false));
            
            rel_diff = norm(y_matvec - y_sparse) / norm(y_sparse);
            fprintf('Matvec vs Sparse Relative Difference: %.3e\n', rel_diff);
            
            testCase.verifyLessThan(rel_diff, 1e-10);
        end
        
        function testMatrixFreePCGSolverAccuracy(testCase)
            % Solve a cantilever problem using matrix-free PCG and compare to direct backslash
            mesh = testCase.mesh;
            brep = testCase.brep;
            
            % Clamp z = 0 face, apply load on z = 4 face
            bc_list = {
                struct('type', 'dirichlet', 'filter', @(x,y,z) abs(z - 0) < 1e-3, ...
                       'value', [0, 0, 0], 'method', 'strong');
                struct('type', 'neumann', 'filter', @(x,y,z) abs(z - 4) < 1e-3, ...
                       'value', [10.0, 0.0, 0.0])
            };
            
            [K_bc, F_bc, bc_state] = apply_boundary_conditions(mesh, brep, bc_list);
            
            % Direct solver
            u_direct = bc_state.solve(K_bc, F_bc);
            
            % Matrix-free PCG solver
            solve_opts.tol = 1e-7;
            solve_opts.max_iter = 1000;
            [u_pcg, info] = solve_octree_gpu(mesh, F_bc, bc_state, solve_opts);
            
            testCase.verifyTrue(info.converged);
            
            % Compare solution vectors
            rel_err = norm(u_pcg - u_direct) / norm(u_direct);
            fprintf('PCG vs Direct Solver Error: %.3e (in %d iterations)\n', rel_err, info.iterations);
            testCase.verifyLessThan(rel_err, 1e-4);
        end
    end
end
