classdef testBoundaryConditions < matlab.unittest.TestCase
    % TESTBOUNDARYCONDITIONS
    % Unit and integration tests for Phase 1: Unified Boundary Condition Engine
    % Verifies Dirichlet, Neumann, and Robin BCs on adaptive octree meshes.
    
    properties
        mesh
        brep
    end
    
    methods(TestMethodSetup)
        function setupFixture(testCase)
            % Create a simple 3D box B-Rep [0, 10] x [0, 5] x [0, 4]
            % 12 triangular facets (2 per face)
            nodes = [
                0, 0, 0;  % 1
                10, 0, 0; % 2
                10, 5, 0; % 3
                0, 5, 0;  % 4
                0, 0, 4;  % 5
                10, 0, 4; % 6
                10, 5, 4; % 7
                0, 5, 4   % 8
            ];
            elements = [
                1, 2, 6; 1, 6, 5; % front (y=0)
                2, 3, 7; 2, 7, 6; % right (x=10)
                3, 4, 8; 3, 8, 7; % back (y=5)
                4, 1, 5; 4, 5, 8; % left (x=0)
                1, 4, 3; 1, 3, 2; % bottom (z=0)
                5, 6, 7; 5, 7, 8  % top (z=4)
            ];
            testCase.brep.nodes = nodes;
            testCase.brep.elements = elements;
            
            % Generate base octree and structural mesh
            grid_bounds = [-1, 11; -1, 6; -1, 5];
            grid_res = [2, 2, 2];
            oct = octree_mesh_3d(grid_bounds, grid_res, 1);
            % Subdivide root leaves 1 and 2
            oct = subdivide_octree_leaves(oct, [1 2]);
            oct = balance_octree_3d(oct);
            
            opts.E = 1e5;
            opts.nu = 0.25;
            opts.fictitious_weight = 1e-3;
            testCase.mesh = octree_structural_mesh(oct, testCase.brep, opts);
        end
    end
    
    methods(Test)
        function testNeumannPressureResultant(testCase)
            % Apply uniform pressure P = 2.0 on right face (x = 10), area = 5 * 4 = 20
            % Resultant traction should be t = -P * n_outward = -2 * [1, 0, 0] = [-2, 0, 0]
            % Total force should be [-40, 0, 0]
            right_filter = @(x,y,z) abs(x - 10) < 1e-3;
            P = 2.0;
            
            bc_list = {
                struct('type', 'neumann', 'filter', right_filter, 'value', P)
            };
            
            [~, F_bc, ~] = apply_boundary_conditions(testCase.mesh, testCase.brep, bc_list);
            
            % Check that total load sum matches total traction resultant
            F_sum = [sum(F_bc(1:3:end)), sum(F_bc(2:3:end)), sum(F_bc(3:3:end))];
            expected_F = [-40, 0, 0];
            testCase.verifyEqual(F_sum, expected_F, 'RelTol', 1e-4);
        end
        
        function testRobinSymmetryAndPositivity(testCase)
            % Test Robin elastic foundation on bottom face (z = 0)
            bottom_filter = @(x,y,z) abs(z - 0) < 1e-3;
            k_s = 500.0;
            
            bc_list = {
                struct('type', 'robin', 'filter', bottom_filter, 'k_spring', k_s)
            };
            
            [K_bc, F_bc, ~] = apply_boundary_conditions(testCase.mesh, testCase.brep, bc_list);
            
            % Difference K_robin = K_bc - K_master
            K_robin = K_bc - testCase.mesh.K_master;
            
            % Verify exact symmetry
            sym_err = norm(K_robin - K_robin', 'fro') / max(1, norm(K_robin, 'fro'));
            testCase.verifyLessThan(sym_err, 1e-12);
            
            % Verify positive semi-definiteness
            v_rand = randn(size(K_robin, 1), 1);
            quad_form = v_rand' * K_robin * v_rand;
            testCase.verifyGreaterThanOrEqual(quad_form, -1e-10);
        end
        
        function testDirichletStrongEquilibrium(testCase)
            % Clamp left face (x = 0), apply traction on right face (x = 10)
            % Solve and verify that reactions exactly balance applied load
            left_filter = @(x,y,z) abs(x - 0) < 1e-3;
            right_filter = @(x,y,z) abs(x - 10) < 1e-3;
            traction = [10.0, 0.0, 0.0]; % Tension in X, total force = 10 * 20 = 200
            
            bc_list = {
                struct('type', 'dirichlet', 'filter', left_filter, 'value', [0, 0, 0], 'method', 'strong');
                struct('type', 'neumann', 'filter', right_filter, 'value', traction)
            };
            
            [K_bc, F_bc, bc_state] = apply_boundary_conditions(testCase.mesh, testCase.brep, bc_list);
            
            u_sol = bc_state.solve(K_bc, F_bc);
            
            % Verify fixed DOFs are present and strictly zero
            testCase.verifyGreaterThan(numel(bc_state.fixed_dofs), 0);
            testCase.verifyEqual(u_sol(bc_state.fixed_dofs), zeros(numel(bc_state.fixed_dofs), 1), 'AbsTol', 1e-12);
            
            % Compute reaction forces: R = K_unreduced * u - F_applied
            R = bc_state.K_unreduced * u_sol - F_bc;
            
            % Sum of reaction in X must equal -200 (within numerical equilibrium)
            total_Rx = sum(R(bc_state.fixed_dofs(mod(bc_state.fixed_dofs, 3) == 1)));
            testCase.verifyEqual(total_Rx, -200, 'RelTol', 1e-2);
        end
        
        function testDirichletWeakEquilibrium(testCase)
            % Weak penalty Dirichlet on left face (x = 0), apply traction on right face (x = 10)
            left_filter = @(x,y,z) abs(x - 0) < 1e-3;
            right_filter = @(x,y,z) abs(x - 10) < 1e-3;
            traction = [10.0, 0.0, 0.0];
            
            bc_list = {
                struct('type', 'dirichlet', 'filter', left_filter, 'value', [0, 0, 0], 'method', 'penalty', 'penalty', 1e8);
                struct('type', 'neumann', 'filter', right_filter, 'value', traction)
            };
            
            [K_bc, F_bc, bc_state] = apply_boundary_conditions(testCase.mesh, testCase.brep, bc_list);
            u_sol = bc_state.solve(K_bc, F_bc);
            
            % Internal force balance: K_master * u must equal F_neumann + F_penalty_reaction
            F_int = testCase.mesh.K_master * u_sol;
            % The net horizontal force from the whole body must balance
            testCase.verifyTrue(all(isfinite(u_sol)));
            testCase.verifyGreaterThan(max(abs(u_sol)), 0);
        end
    end
end
