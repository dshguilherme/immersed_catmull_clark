function mesh = importMSH(filename)
% IMPORTMSH Import a Gmsh .msh file using the unified Python/Gmsh bridge.
% Returns struct with .nodes and .elements.
if nargin < 1 || isempty(filename)
    error('IMPORTMSH:MissingInput', 'Filename must be provided.');
end
mesh = importMeshViaPython(filename);
end
