classdef testGPUAssemblyContact < matlab.unittest.TestCase
    % TESTGPUASSEMBLYCONTACT
    % Tests matrix-free GPU assembly matvec and GPU PCG contact solver integration
    
    properties
        assembly
    end
    
    methods (TestMethodSetup)
        function setupTestAssembly(testCase)
            bA.nodes = [0 0 0; 2 0 0; 2 2 0; 0 2 0; 0 0 2; 2 0 2; 2 2 2; 0 2 2];
            bA.elements = [1 2 6; 1 6 5; 2 3 7; 2 7 6; 3 4 8; 3 8 7; 4 1 5; 4 5 8; 1 4 3; 1 3 2; 5 6 7; 5 7 8];
            
            bB.nodes = [0 0 2; 2 0 2; 2 2 2; 0 2 2; 0 0 4; 2 0 4; 2 2 4; 0 2 4];
            bB.elements = [1 2 6; 1 6 5; 2 3 7; 2 7 6; 3 4 8; 3 8 7; 4 1 5; 4 5 8; 1 4 3; 1 3 2; 5 6 7; 5 7 8];
            
            bodies = {struct('brep', bA, 'name', 'BlockA', 'grid_res', [2 2 2], 'max_level', 0), ...
                      struct('brep', bB, 'name', 'BlockB', 'grid_res', [2 2 2], 'max_level', 0)};
            
            testCase.assembly = setup_assembly_3d(bodies, struct('gap_tol', 0.1));
        end
    end
    
    methods (Test)
        function testAssemblyMatvecConsistency(testCase)
            ass = testCase.assembly;
            p = randn(ass.total_dof, 1);
            
            % Direct K * p
            y_direct = ass.K_assembly * p;
            
            % Matrix-free matvec
            y_mf = gpu_assembly_matvec(ass, p, [], [], struct('use_gpu', false));
            
            err = norm(y_direct - y_mf) / norm(y_direct);
            testCase.verifyLessThan(err, 1e-12, 'GPU assembly matvec must match direct assembly matrix');
        end
        
        function testBondedContactGPUPCG(testCase)
            % Body 1: Base block [0, 10] x [0, 10] x [0, 4]
            n1 = [0, 0, 0; 10, 0, 0; 10, 10, 0; 0, 10, 0; 0, 0, 4; 10, 0, 4; 10, 10, 4; 0, 10, 4];
            e1 = [1 2 6; 1 6 5; 2 3 7; 2 7 6; 3 4 8; 3 8 7; 4 1 5; 4 5 8; 1 4 3; 1 3 2; 5 6 7; 5 7 8];
            b1.brep.nodes = n1; b1.brep.elements = e1;
            b1.name = 'BaseBlock'; b1.grid_res = [2, 2, 2]; b1.max_level = 0;
            
            % Body 2: Punch block [2, 8] x [2, 8] x [4, 8]
            n2 = [2, 2, 4; 8, 2, 4; 8, 8, 4; 2, 8, 4; 2, 2, 8; 8, 2, 8; 8, 8, 8; 2, 8, 8];
            e2 = [1 2 6; 1 6 5; 2 3 7; 2 7 6; 3 4 8; 3 8 7; 4 1 5; 4 5 8; 1 4 3; 1 3 2; 5 6 7; 5 7 8];
            b2.brep.nodes = n2; b2.brep.elements = e2;
            b2.name = 'PunchBlock'; b2.grid_res = [2, 2, 2]; b2.max_level = 0;
            
            ass = setup_assembly_3d({b1, b2}, struct('gap_tol', 0.5));
            
            bc_spec = {
                struct('body_idx', 1, 'type', 'dirichlet', 'filter', @(x,y,z) abs(z - 0) < 1e-3, ...
                       'value', [0, 0, 0], 'method', 'strong');
                struct('body_idx', 2, 'type', 'neumann', 'filter', @(x,y,z) abs(z - 8) < 1e-3, ...
                       'value', [0, 0, -10.0])
            };
            
            % Direct solve
            [u_dir, res_dir] = solve_assembly_contact_3d(ass, zeros(ass.total_dof, 1), bc_spec, ...
                struct('mode', 'bonded', 'linear_solver', 'direct'));
            
            % GPU PCG solve
            [u_gpu, res_gpu] = solve_assembly_contact_3d(ass, zeros(ass.total_dof, 1), bc_spec, ...
                struct('mode', 'bonded', 'linear_solver', 'gpu_pcg', 'pcg_tol', 1e-6, 'pcg_max_iter', 2000));
            
            testCase.verifyTrue(res_dir.converged);
            testCase.verifyTrue(res_gpu.converged);
            
            rel_diff = norm(u_dir - u_gpu) / norm(u_dir);
            testCase.verifyLessThan(rel_diff, 5e-3, 'GPU PCG solve must match direct bonded contact solution');
        end
    end
end
