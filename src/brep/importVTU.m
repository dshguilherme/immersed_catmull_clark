function mesh = importVTU(filename)
% IMPORTVTU Import a VTU/VTK mesh file using the unified bridge.
% Returns struct with .nodes and .elements.
if nargin < 1 || isempty(filename)
    error('IMPORTVTU:MissingInput', 'Filename must be provided.');
end
mesh = importMeshViaPython(filename);
end
