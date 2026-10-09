% TEST_BREP_IMPORT
% Verifies B-Rep importing across supported CAD formats
this_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(this_dir, '..', 'src')));

step_path = fullfile(this_dir, '..', 'Models', 'NIST-PMI-STEP-Files', 'NIST-PMI-STEP-Files', 'AP203 geometry only', 'nist_ctc_01_asme1_rd.stp');

fprintf('Testing STEP import via importBRep...\n');
mesh = importBRep(step_path);

assert(~isempty(mesh.nodes), 'Nodes array should not be empty');
assert(~isempty(mesh.elements), 'Elements array should not be empty');
assert(size(mesh.nodes, 2) == 3, 'Nodes should be 3D');
assert(size(mesh.elements, 2) == 3, 'Boundary elements should be triangles');

fprintf('STEP import passed! Vertices: %d, Faces: %d\n', size(mesh.nodes, 1), size(mesh.elements, 1));
