classdef testImportVTU < matlab.unittest.TestCase
    methods(Test)
        function testMissingInput(testCase)
            % Verify that calling importVTU with empty filename throws error
            testCase.verifyError(@() importVTU(''), 'IMPORTVTU:MissingInput');
        end
    end
end
