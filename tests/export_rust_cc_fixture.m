function export_rust_cc_fixture()
% EXPORT_RUST_CC_FIXTURE  MATLAB reference for the Rust port of src/catmull_clark.
%   Two Catmull-Clark steps on a perturbed 3x2 quad grid, centroid triangle
%   subdivision, barycentric basis, projection and subdivision matrices.
%   Writes rust/tests/fixtures/catmull_clark.json.

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', 'src')));
out = fullfile(here, '..', 'rust', 'tests', 'fixtures', 'catmull_clark.json');
[X, Y] = ndgrid(0:3, 0:2);
V = [X(:), Y(:), 0.1 * sin(X(:) + 2 * Y(:))];
V(6, :) = V(6, :) + [0.2 -0.1 0.3];
id = @(i, j) i + 4 * j + 1;
F = [];
for j = 0:1
    for i = 0:2
        F = [F; id(i, j), id(i+1, j), id(i+1, j+1), id(i, j+1)]; %#ok<AGROW>
    end
end
[V1, F1] = subdivide_quad_catmull_clark(V, F);
[V2, F2] = subdivide_quad_catmull_clark(V1, F1);
Vt = [0 0 0; 1 0 0; 0 1 0; 1 1 0.5]; Ft = [1 2 3; 2 4 3];
[Vt2, Ft2] = subdivide(Vt, Ft);
P = [0.2 0.2 0; 0.7 0.6 0.25; 2 2 2];
B = basisFunctions(Vt, Ft, P);
[P1, P2] = catmull_clark_projection_matrices(5, 3);
[Q1, ~, ~] = catmull_clark_projection_matrices_3d(4, 3, 2, 2);
[Psub, P3d] = catmull_clark_subdivision_matrix_1d();
data.V = V; data.F = F - 1;
data.V1 = V1; data.F1 = F1 - 1; data.V2 = V2; data.F2 = F2 - 1;
data.Vt = Vt; data.Ft = Ft - 1; data.Vt2 = Vt2; data.Ft2 = Ft2 - 1; data.P = P; data.B = B;
data.P1 = P1; data.P2 = P2; data.Q1 = Q1; data.Psub = Psub; data.P3d_sum = sum(P3d(:)); data.P3d_size = size(P3d);
fid = fopen(out, 'w'); fwrite(fid, jsonencode(data), 'char'); fclose(fid);
fprintf('Wrote %s (%d -> %d -> %d quads)\n', out, size(F, 1), size(F1, 1), size(F2, 1));
end
