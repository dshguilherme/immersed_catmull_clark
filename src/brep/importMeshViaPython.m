function mesh = importMeshViaPython(filename)
% IMPORTMESHVIAPYTHON Unified importer for STEP, MSH, VTU, and VTK models
% via the Gmsh / Python bridge.
%
% Returns a struct:
%   mesh.nodes    - N x 3 double array of vertex coordinates
%   mesh.elements - M x 3 int connectivity matrix of triangular boundary facets
%   mesh.tags     - Face / boundary IDs

[~, fname, ~] = fileparts(filename);
temp_obj = fullfile(tempdir, sprintf('%s_%d_brep.obj', fname, randi(1000000)));

% Python with gmsh + numpy: IMMERSED_IGA_PYTHON if set, otherwise 'python' on the PATH
python_exe = getenv('IMMERSED_IGA_PYTHON');
if isempty(python_exe), python_exe = 'python'; end
script_py = fullfile(fileparts(mfilename('fullpath')), 'brep_to_tri_mesh.py');

cmd = sprintf('"%s" "%s" "%s" "%s"', python_exe, script_py, filename, temp_obj);
[status, out] = system(cmd);

if status ~= 0
    error('importMeshViaPython:BridgeError', 'Failed to parse model via Python backend:\n%s', out);
end

% Read exported OBJ file into MATLAB mesh struct
fid = fopen(temp_obj, 'r');
if fid == -1
    error('importMeshViaPython:FileError', 'Could not open temporary mesh file %s', temp_obj);
end

nodes = [];
faces = [];

while ~feof(fid)
    line = strtrim(fgetl(fid));
    if startsWith(line, 'v ')
        parts = sscanf(line(3:end), '%f %f %f');
        nodes = [nodes; parts(:)'];
    elseif startsWith(line, 'f ')
        parts = sscanf(line(3:end), '%d %d %d');
        faces = [faces; parts(:)'];
    end
end
fclose(fid);
if exist(temp_obj, 'file'), delete(temp_obj); end

mesh.nodes = nodes;
mesh.elements = faces;
mesh.tags = ones(size(faces, 1), 1);

fprintf('Imported Model "%s": %d surface vertices, %d boundary faces.\n', ...
    fname, size(mesh.nodes, 1), size(mesh.elements, 1));

end
