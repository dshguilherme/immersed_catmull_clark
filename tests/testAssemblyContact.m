classdef testAssemblyContact < matlab.unittest.TestCase
    % TESTASSEMBLYCONTACT
    % Unit and integration tests for Phase 2: Multi-Body Assembly & Unilateral Contact
    % Verifies assembly setup, bonded contact, unilateral active-set Newton, and separation.
    
    properties
        assembly
        brep_bottom
        brep_top
    end
    
    methods(TestMethodSetup)
        function setupFixture(testCase)
            % Body 1: Bottom block [0, 10] x [0, 10] x [0, 4]
            n1 = [
                0, 0, 0; 10, 0, 0; 10, 10, 0; 0, 10, 0;
                0, 0, 4; 10, 0, 4; 10, 10, 4; 0, 10, 4
            ];
            e1 = [
                1, 2, 6; 1, 6, 5; % front
                2, 3, 7; 2, 7, 6; % right
                3, 4, 8; 3, 8, 7; % back
                4, 1, 5; 4, 5, 8; % left
                1, 4, 3; 1, 3, 2; % bottom (z=0)
                5, 6, 7; 5, 7, 8  % top (z=4)
            ];
            testCase.brep_bottom.nodes = n1;
            testCase.brep_bottom.elements = e1;
            
            % Body 2: Top block [2, 8] x [2, 8] x [4, 8]
            % Interface at z = 4
            n2 = [
                2, 2, 4; 8, 2, 4; 8, 8, 4; 2, 8, 4;
                2, 2, 8; 8, 2, 8; 8, 8, 8; 2, 8, 8
            ];
            e2 = [
                1, 2, 6; 1, 6, 5; % front
                2, 3, 7; 2, 7, 6; % right
                3, 4, 8; 3, 8, 7; % back
                4, 1, 5; 4, 5, 8; % left
                1, 4, 3; 1, 3, 2; % bottom (z=4) - contact with body 1
                5, 6, 7; 5, 7, 8  % top (z=8)
            ];
            testCase.brep_top.nodes = n2;
            testCase.brep_top.elements = e2;
            
            b1.brep = testCase.brep_bottom;
            b1.name = 'BaseBlock';
            b1.grid_res = [2, 2, 2];
            b1.max_level = 0;
            
            b2.brep = testCase.brep_top;
            b2.name = 'PunchBlock';
            b2.grid_res = [2, 2, 2];
            b2.max_level = 0;
            
            opts.gap_tol = 0.5;
            testCase.assembly = setup_assembly_3d({b1, b2}, opts);
        end
    end
    
    methods(Test)
        function testAssemblyDetection(testCase)
            % Verify that the assembly detected 2 bodies and contact interface
            testCase.verifyEqual(testCase.assembly.n_bodies, 2);
            testCase.verifyGreaterThan(testCase.assembly.total_dof, 0);
            testCase.verifyEqual(numel(testCase.assembly.interfaces), 1);
            
            inter = testCase.assembly.interfaces{1};
            testCase.verifyEqual(inter.body_A, 1);
            testCase.verifyEqual(inter.body_B, 2);
            testCase.verifyGreaterThan(inter.n_pairs, 0);
        end
        
        function testBondedContactTransmission(testCase)
            % Clamp bottom face of BaseBlock (z = 0), apply traction on top of PunchBlock (z = 8)
            % Under bonded contact, force must transmit to the base
            bc_spec = {
                struct('body_idx', 1, 'type', 'dirichlet', 'filter', @(x,y,z) abs(z - 0) < 1e-3, ...
                       'value', [0, 0, 0], 'method', 'strong');
                struct('body_idx', 2, 'type', 'neumann', 'filter', @(x,y,z) abs(z - 8) < 1e-3, ...
                       'value', [0, 0, -10.0]) % Pushing down
            };
            
            F_ext = zeros(testCase.assembly.total_dof, 1);
            opts.mode = 'bonded';
            opts.gamma_c = 100.0;
            
            [u_sol, res] = solve_assembly_contact_3d(testCase.assembly, F_ext, bc_spec, opts);
            
            testCase.verifyTrue(res.converged);
            % Check that punch has downward displacement (u_z < 0)
            range2 = testCase.assembly.bodies{2}.dof_range;
            u_z_punch = u_sol(range2(3:3:end));
            testCase.verifyLessThan(mean(u_z_punch), 0.0);
            
            % Base should also experience compression
            range1 = testCase.assembly.bodies{1}.dof_range;
            u_z_base = u_sol(range1(3:3:end));
            testCase.verifyLessThan(min(u_z_base), 0.0);
        end
        
        function testUnilateralContactCompression(testCase)
            % Under downward compression, active set should engage and prevent penetration
            bc_spec = {
                struct('body_idx', 1, 'type', 'dirichlet', 'filter', @(x,y,z) abs(z - 0) < 1e-3, ...
                       'value', [0, 0, 0], 'method', 'strong');
                struct('body_idx', 2, 'type', 'neumann', 'filter', @(x,y,z) abs(z - 8) < 1e-3, ...
                       'value', [0, 0, -20.0]) % Downward compression
            };
            
            F_ext = zeros(testCase.assembly.total_dof, 1);
            opts.mode = 'unilateral';
            opts.gamma_c = 100.0;
            opts.max_iter = 10;
            
            [u_sol, res] = solve_assembly_contact_3d(testCase.assembly, F_ext, bc_spec, opts);
            
            testCase.verifyTrue(res.converged);
            testCase.verifyGreaterThan(res.n_active, 0);
            % Non-penetration: min_gap should not be excessively negative
            testCase.verifyGreaterThanOrEqual(res.min_gap, -0.05);
        end
        
        function testUnilateralSeparation(testCase)
            % Under upward tension, contact should disengage (0 active pairs)
            bc_spec = {
                struct('body_idx', 1, 'type', 'dirichlet', 'filter', @(x,y,z) abs(z - 0) < 1e-3, ...
                       'value', [0, 0, 0], 'method', 'strong');
                struct('body_idx', 2, 'type', 'neumann', 'filter', @(x,y,z) abs(z - 8) < 1e-3, ...
                       'value', [0, 0, +20.0]) % Pulling UP away from base
            };
            
            F_ext = zeros(testCase.assembly.total_dof, 1);
            opts.mode = 'unilateral';
            opts.gamma_c = 100.0;
            opts.max_iter = 10;
            
            [u_sol, res] = solve_assembly_contact_3d(testCase.assembly, F_ext, bc_spec, opts);
            
            testCase.verifyTrue(res.converged);
            % Contact pairs must NOT be active when pulling apart!
            testCase.verifyEqual(res.n_active, 0);
            
            % Punch moves upward
            range2 = testCase.assembly.bodies{2}.dof_range;
            u_z_punch = u_sol(range2(3:3:end));
            testCase.verifyGreaterThan(mean(u_z_punch), 0.0);
        end
    end
end
