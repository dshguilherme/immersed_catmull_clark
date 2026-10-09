function [K, t] = wq_form(S, xPhys, density_type, penal, Emin, symmetrize, output)
% WQ_FORM Per-iteration batched weighted-quadrature stiffness formation (2D).
% Uses the density-independent arrays prepared by WQ_SETUP. All row-wise
% sum-factorization contractions are executed as two batched page products
% (pagemtimes), so the GPU receives a few large kernels instead of
% O(N_dof) tiny ones.
%
%   density_type - 'element' (xPhys is [nelx x nely]) or 'spline' (control net)
%   output       - 'cpu_sparse' (default): gather values, build sparse K on host
%                  'device_sparse': build sparse K on the device (no host transfer)
%                  'values'       : return the raw values only (no sparse build)
%
% Timing breakdown t (seconds, device synchronized at each boundary):
%   t.upload   - host->device copy of the density array
%   t.kernel   - SIMP coefficient + all sum-factorization contractions
%   t.extract  - extraction of the nonzero pattern from the padded pages
%   t.download - device->host copy of the nonzero values
%   t.sparse   - sparse matrix construction (+ symmetrization)

if nargin < 3 || isempty(density_type), density_type = 'element'; end
if nargin < 4 || isempty(penal), penal = 3; end
if nargin < 5 || isempty(Emin), Emin = 1e-9; end
if nargin < 6 || isempty(symmetrize), symmetrize = true; end
if nargin < 7 || isempty(output), output = 'cpu_sparse'; end

on_gpu = strcmp(S.device, 'gpu');
sync = @() [];
if on_gpu
    dev = gpuDevice;
    sync = @() wait(dev);
end

%% Upload density
sync(); t0 = tic;
x = cast(xPhys, S.precision);
if on_gpu, x = gpuArray(x); end
sync(); t.upload = toc(t0);

%% Kernel: SIMP coefficient at WQ points + batched sum factorization
t0 = tic;
if strcmpi(density_type, 'element')
    rho_q = x(S.e1, S.e2);
else
    rho_q = S.S1 * x * S.S2.';
end
simp = Emin + (rho_q .^ penal) * (1 - Emin);
s = simp(S.LIN);                                     % [nq1 x nq2 x N]

V = [];
for k1 = 1:2
    for k2 = 1:2
        T = S.C0p{k1, k2} .* s;                       % [nq1 x nq2 x N x 4]
        R = pagemtimes(T, 'none', S.WBe{2, S.typ(k1, k2, 2)}, 'transpose');   % [nq1 x nn2 x N x 4]
        R = pagemtimes(S.WBe{1, S.typ(k1, k2, 1)}, R);                       % [nn1 x nn2 x N x 4]
        if isempty(V), V = R; else, V = V + R; end
    end
end
sync(); t.kernel = toc(t0);

%% Extract nonzeros (same ordering as the CPU row loop)
t0 = tic;
nzb = S.nnz_block;
vals = zeros(4 * nzb, 1, 'like', V);
for c = 1:4
    Vc = V(:, :, :, c);
    vals((c-1)*nzb + 1 : c*nzb) = Vc(S.mask);
end
sync(); t.extract = toc(t0);

%% Output
N = S.N;
r = S.rows; cc = S.cols;
R_all = [r; r + N; r; r + N];          % blocks (1,1),(2,1),(1,2),(2,2)
C_all = [cc; cc; cc + N; cc + N];
switch output
    case 'values'
        t0 = tic;
        if on_gpu, vals = gather(vals); end
        t.download = toc(t0);
        t.sparse = 0;
        K = vals;
    case 'device_sparse'
        t.download = 0;
        t0 = tic;
        if on_gpu
            K = sparse(gpuArray(R_all), gpuArray(C_all), double(vals), 2*N, 2*N);
        else
            K = sparse(R_all, C_all, double(vals), 2*N, 2*N);
        end
        if symmetrize, K = 0.5 * (K + K.'); end
        sync(); t.sparse = toc(t0);
    otherwise % 'cpu_sparse'
        t0 = tic;
        if on_gpu, vals = gather(vals); end
        t.download = toc(t0);
        t0 = tic;
        K = sparse(R_all, C_all, double(vals), 2*N, 2*N);
        if symmetrize, K = 0.5 * (K + K.'); end
        t.sparse = toc(t0);
end
t.total = t.upload + t.kernel + t.extract + t.download + t.sparse;
end
