classdef testObstacleCourse < matlab.unittest.TestCase
    % TESTOBSTACLECOURSE
    % Automated test runner executing the 5-obstacle publication benchmark course.
    
    methods(Test)
        function testFiveObstacles(testCase)
            this_dir = fileparts(mfilename('fullpath'));
            addpath(fullfile(this_dir, '..', 'benchmarks'));
            
            res = run_complete_obstacle_course();
            
            testCase.verifyTrue(res.obstacle1.passed, 'Obstacle 1: Patch test failed');
            testCase.verifyTrue(res.obstacle2.passed, 'Obstacle 2: Hertzian contact failed');
            testCase.verifyTrue(res.obstacle3.passed, 'Obstacle 3: AMR singular riser failed');
            testCase.verifyTrue(res.obstacle4.passed, 'Obstacle 4: NIST multi-body assembly failed');
            testCase.verifyTrue(res.obstacle5.passed, 'Obstacle 5: GPU matrix-free solver failed');
            testCase.verifyTrue(res.all_passed, 'Overall obstacle course failed');
        end
    end
end
