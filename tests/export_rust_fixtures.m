function export_rust_fixtures()
% EXPORT_RUST_FIXTURES  Write MATLAB reference data used by the Rust test suite.
%   Produces rust/tests/fixtures/iga_kernel.json with, for several 2D/3D box spaces:
%   connectivity, ||K||_F, K*v for fixed probe vectors and the load vector of a
%   smooth body force. The Rust tests (rust/tests/iga_vs_matlab.rs) rebuild the
%   same quantities and compare. Re-run after changing src/iga.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'src', 'iga'));
out = fullfile(here, '..', 'rust', 'tests', 'fixtures', 'iga_kernel.json');
if ~exist(fileparts(out), 'dir'), mkdir(fileparts(out)); end

[lambda, mu] = deal(0.3 / (1.3 * 0.4), 1 / 2.6);   % E = 1, nu = 0.3
cases = {};
for p = 1:3
    cases{end+1} = struct('bounds', [0.5 2.5; -1 0.2], 'nsub', [5 4], 'degree', p); %#ok<AGROW>
end
for p = 1:3
    cases{end+1} = struct('bounds', [0.5 2.5; -1 0.2; 0.1 0.4], 'nsub', [4 3 3], 'degree', p); %#ok<AGROW>
end

records = cell(1, numel(cases));
for k = 1:numel(cases)
    c = cases{k};
    sp = iga_space_box(c.bounds, c.nsub, c.degree);
    [Ke, tid] = iga_elasticity_element_matrices(sp, lambda, mu);
    [r, cc] = iga_element_rows_cols(sp.connectivity);
    K = sparse(r, cc, reshape(Ke(:, :, tid), [], 1), sp.ndof, sp.ndof);
    i = (1:sp.ndof).';
    probes = zeros(sp.ndof, 3);
    for j = 1:3
        probes(:, j) = K * sin(0.37 * i * j);
    end
    if sp.dim == 2
        f = @(x, y) cat(1, reshape(x .* y, [1 size(x)]), reshape(sin(x) + y.^2, [1 size(x)]));
    else
        f = @(x, y, z) cat(1, reshape(x .* y, [1 size(x)]), reshape(sin(x) + z, [1 size(x)]), reshape(y.^2, [1 size(x)]));
    end
    F = iga_load_vector(sp, f);
    rec.bounds = c.bounds;
    rec.nsub = c.nsub;
    rec.degree = c.degree;
    rec.ndof = sp.ndof;
    rec.ntypes = size(Ke, 3);
    rec.connectivity = sp.connectivity(:).' - 1;     % 0-based, element-major
    rec.frobenius = norm(K, 'fro');
    rec.probes = probes;                               % [ndof x 3]
    rec.load = F.';
    records{k} = rec;
end
data.lambda = lambda;
data.mu = mu;
data.probe_definition = 'v_j(i) = sin(0.37 * i * j), i = 1..ndof (1-based), j = 1..3';
data.cases = records;
fid = fopen(out, 'w');
fwrite(fid, jsonencode(data, 'PrettyPrint', false), 'char');
fclose(fid);
fprintf('Wrote %s\n', out);
end
