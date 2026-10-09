classdef testSubdivide < matlab.unittest.TestCase
    methods(Test)
        function testSimpleTriangle(testCase)
            % Create a single triangle mesh
            V = [0 0 0; 1 0 0; 0 1 0];
            F = [1 2 3];
            [Vnew, Fnew] = subdivide(V, F);
            % Expect number of vertices = original + number of faces
            testCase.verifySize(Vnew, [size(V,1)+size(F,1), 3]);
            % Expect number of faces = 3 * original faces
            testCase.verifySize(Fnew, [3*size(F,1), 3]);
        end
    end
end
