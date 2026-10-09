function mesh = importSTEP(filename)
% IMPORTSTEP Load a STEP geometry file (.stp / .step) using Gmsh backend.
% Returns a unified triangular surface mesh struct:
%   mesh.nodes    - N x 3 double array of vertex coordinates
%   mesh.elements - M x 3 int connectivity matrix of triangular boundary facets
%   mesh.tags     - Face IDs

[fpath, fname, ~] = fileparts(filename);
temp_obj = fullfile(tempdir, [fname, '_brep.obj']);

python_exe = 'C:\Users\dshgu\miniconda3\python.exe';
script_py = fullfile(fileparts(mfilename('fullpath')), 'step_to_tri_mesh.py');

cmd = sprintf('"%s" "%s" "%s" "%s"', python_exe, script_py, filename, temp_obj);
[status, out] = system(cmd);

if status ~= 0
    error('importSTEP:GmshError', 'Failed to parse STEP file via Gmsh:\n%s', out);
end

% Read exported OBJ file into MATLAB mesh struct
fid = fopen(temp_obj, 'r');
if fid == -1
    error('importSTEP:FileError', 'Could not open temporary mesh file %s', temp_obj);
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
delete(temp_obj);

mesh.nodes = nodes;
mesh.elements = faces;
mesh.tags = ones(size(faces, 1), 1);

fprintf('Imported STEP Model "%s": %d surface vertices, %d boundary faces.\n', ...
    fname, size(mesh.nodes, 1), size(mesh.elements, 1));

end
