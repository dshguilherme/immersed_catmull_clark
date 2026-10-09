% RUN_ALL_TESTS
% Central test runner for the Immersed Catmull-Clark IGA stack
this_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(this_dir, '..', 'src')));
addpath(this_dir);

fprintf('========================================================================\n');
fprintf('  RUNNING IMMERSED CATMULL-CLARK IGA TEST SUITE\n');
fprintf('========================================================================\n\n');

suite = testsuite(this_dir);
results = run(suite);

n_fail = sum([results.Failed]);
n_pass = sum([results.Passed]);

fprintf('\nTest Results Summary: %d Passed, %d Failed\n', n_pass, n_fail);
if n_fail > 0
    error('Test suite failed with %d failure(s).', n_fail);
else
    fprintf('ALL TESTS PASSED SUCCESSFULLY.\n');
end
