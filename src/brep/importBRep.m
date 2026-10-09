function mesh = importBRep(filename)
% IMPORTBREP Load a B‑Rep model (STP, STEP, VTK, VTU, MSH, STL) into a unified mesh struct.
%   mesh = importBRep(filename) reads the file, detects its extension,
%   and returns a struct with fields:
%       mesh.nodes    – Nx3 double array of vertex coordinates
%       mesh.elements – Mx3 connectivity of triangular boundary facets
%       mesh.tags     – optional integer tags / physical groups

if nargin < 1 || isempty(filename)
    error('IMPORTBREP:MissingInput', 'Filename must be provided.');
end

if ~exist(filename, 'file')
    error('IMPORTBREP:FileNotFound', 'File not found: %s', filename);
end

[~,~,ext] = fileparts(lower(filename));
switch ext
    case {'.stp', '.step', '.msh', '.vtk', '.vtu', '.stl'}
        mesh = importMeshViaPython(filename);
    otherwise
        error('Unsupported B‑Rep extension: %s', ext);
end

end
