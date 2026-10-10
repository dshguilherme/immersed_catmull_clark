function export_rust_wq_fixture()
% EXPORT_RUST_WQ_FIXTURE  MATLAB reference for the Rust weighted-quadrature (FastFormation)
%   assembly, sensitivities and 2D topology optimization. Writes
%   rust/tests/fixtures/wq_fastformation.json.

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', 'src')));
out = fullfile(here, '..', 'rust', 'tests', 'fixtures', 'wq_fastformation.json');
set(0, 'DefaultFigureVisible', 'off');
rng(7);
nelx = 8; nely = 4; p = 3;
xe = 0.2 + 0.8 * rand(nelx, nely);
xs = 0.2 + 0.8 * rand(nelx + p, nely + p);
U = randn(2 * (nelx + p) * (nely + p), 1);
xe3 = 0.2 + 0.8 * rand(4, 3, 2);
sp = iga_space_box([0 1; 0 0.5], [nelx nely], p);
sp3 = iga_space_box([0 1.2; 0 0.6; 0 0.3], [4 3 2], 2);
Kfe = fast_stiffness_assembly(sp, 1.0, 0.3, xe, 'element', 3, 1e-3, true);
Kfs = fast_stiffness_assembly(sp, 1.0, 0.3, xs, 'spline', 3, 1e-3, true);
Kf3 = fast_stiffness_assembly(sp3, 1.0, 0.3, xe3, 'element', 3, 1e-3, true);
Q = iga_wq_rules_1d(sp.knots{1}, p);            % interior layout (default)
Qc = iga_wq_rules_1d(sp.knots{1}, p, 'calabro');
Kfe_cal = fast_stiffness_assembly(sp, 1.0, 0.3, xe, 'element', 3, 1e-3, true, sp, 'calabro');
probe = @(K) (K * sin(0.37 * (1:size(K, 1)).')).';
data.nel = [nelx nely]; data.degree = p; data.xe = xe(:).'; data.xs = xs(:).'; data.U = U.'; data.xe3 = xe3(:).';
data.Kfe_probe = probe(Kfe); data.Kfs_probe = probe(Kfs); data.Kf3_probe = probe(Kf3);
data.Kfe_fro = norm(Kfe, 'fro'); data.Kfs_fro = norm(Kfs, 'fro'); data.Kf3_fro = norm(Kf3, 'fro');
data.dCe = reshape(fast_sensitivities(U, sp, xe, 'element', 3, 1e-3, 1.0, 0.3), 1, []);
data.dCs = reshape(fast_sensitivities(U, sp, xs, 'spline', 3, 1e-3, 1.0, 0.3), 1, []);
data.wq_points = Q.all_points;
data.wq_w11_first = Q.quad_weights_11{1}; data.wq_w10_mid = Q.quad_weights_10{5};
data.wqc_points = Qc.all_points;
data.wqc_w11_first = Qc.quad_weights_11{1}; data.wqc_w10_mid = Qc.quad_weights_10{5};
data.Kfe_cal_probe = probe(Kfe_cal);
old = cd(tempdir); c = onCleanup(@() cd(old));
[x2e, c2e] = topopt_iga_fast(16, 8, 0.5, 3, 2, 5, 'element', 3);
[x2s, c2s] = topopt_iga_fast(16, 8, 0.5, 3, 2, 5, 'spline', 3);
close all force;
data.topo_e_c = c2e(1:5).'; data.topo_e_x = x2e(:).';
data.topo_s_c = c2s(1:5).'; data.topo_s_x = x2s(:).';
fid = fopen(out, 'w'); fwrite(fid, jsonencode(data), 'char'); fclose(fid);
fprintf('Wrote %s\n', out);
end
