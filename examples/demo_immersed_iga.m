% demo_immersed_iga.m - Demonstration of Immersed IGA workflow
% This script loads a sample B‑Rep model, performs one Catmull‑Clark
% subdivision step, and calls the GPU operator.

% Adjust the path below to point to a valid STEP/VTU/MSH file on your
% system.
sampleFile = fullfile('..', 'samples', 'sample.stp'); % placeholder

% Load the geometry/mesh
mesh = importBRep(sampleFile);

% Simple check that the mesh has triangular elements
if ~isfield(mesh, 'elements') || size(mesh.elements,2) ~= 3
    error('Demo expects a triangular mesh (elements Nx3).');
end

% Perform one Catmull‑Clark (simple) subdivision step
[Vnew, Fnew] = subdivide(mesh.nodes, mesh.elements);

% Update mesh struct with refined geometry
refinedMesh.nodes = Vnew;
refinedMesh.elements = Fnew;

% Apply the GPU operator (e.g., matrix‑free weighted quadrature)
% The function gpuApply is assumed to take a mesh struct and return a
% result struct. Adjust as needed for your actual implementation.
result = gpuApply(refinedMesh);

% Display a brief summary of the result
fprintf('GPU operator completed.\n');
if isfield(result, 'info')
    disp(result.info);
end
