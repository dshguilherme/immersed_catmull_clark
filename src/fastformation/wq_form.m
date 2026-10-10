function [K, t] = wq_form(S, xPhys, density_type, penal, Emin, symmetrize, output)
% WQ_FORM Per-iteration batched weighted-quadrature stiffness formation (2D/3D).
% Uses the density-independent arrays prepared by WQ_SETUP. For every derivative
% pair (k1, k2) the SIMP factor on each row's local WQ grid is contracted direction
% by direction with batched page products (pagemtimes), then scaled by the constant
% coefficients c(i,j,k1,k2) into the d^2 displacement blocks.
%
%   density_type - 'element' (xPhys has nel_dir shape) or 'spline' (control net)
%   output       - 'cpu_sparse' (default): gather values, build sparse K on host
%                  'device_sparse': build sparse K on the device (no host transfer)
%                  'values'       : return the raw values only (no sparse build)
%
% Timing breakdown t (seconds, device synchronized at each boundary):
%   t.upload, t.kernel (SIMP + all contractions), t.extract, t.download, t.sparse

if nargin < 3 || isempty(density_type), density_type = 'element'; end
if nargin < 4 || isempty(penal), penal = 3; end
if nargin < 5 || isempty(Emin), Emin = 1e-9; end
if nargin < 6 || isempty(symmetrize), symmetrize = true; end
if nargin < 7 || isempty(output), output = 'cpu_sparse'; end
on_gpu = strcmp(S.device, 'gpu');
sync = @() [];
if on_gpu, dev = gpuDevice; sync = @() wait(dev); end
nsd = S.dim; N = S.N;

%% Upload density
sync(); t0 = tic;
x = cast(xPhys, S.precision);
if on_gpu, x = gpuArray(x); end
sync(); t.upload = toc(t0);

%% Kernel
t0 = tic;
if strcmpi(density_type, 'element')
    if nsd == 2, rho_q = x(S.ebin{1}, S.ebin{2}); else, rho_q = x(S.ebin{1}, S.ebin{2}, S.ebin{3}); end
else
    rho_q = x;
    for d = 1:nsd
        rho_q = mode_product(S.Sd{d}, rho_q, d, nsd);
    end
end
simp = Emin + (rho_q .^ penal) * (1 - Emin);
sl = simp(S.LIN);                                   % [nq1 x .. x nqd x N]
nb = S.nnmax; nq = S.nqmax;
V = cell(nsd, nsd);
for k1 = 1:nsd
    for k2 = 1:nsd
        R = contract(sl, S.WBe, squeeze(S.typ(k1, k2, :)), nb, nq, N, nsd);
        for i = 1:nsd
            for j = 1:nsd
                cij = S.c(i, j, k1, k2);
                if cij ~= 0
                    if isempty(V{i, j}), V{i, j} = cij * R; else, V{i, j} = V{i, j} + cij * R; end
                end
            end
        end
    end
end
sync(); t.kernel = toc(t0);

%% Extract nonzeros (block order: i fastest, then j)
t0 = tic;
nzb = S.nnz_block;
vals = zeros(nsd^2 * nzb, 1, 'like', sl);
blk = 0;
for j = 1:nsd
    for i = 1:nsd
        blk = blk + 1;
        vals((blk-1)*nzb + 1 : blk*nzb) = V{i, j}(S.mask);
    end
end
sync(); t.extract = toc(t0);

%% Output
r = S.rows; cc = S.cols;
R_all = zeros(nsd^2 * nzb, 1); C_all = R_all; blk = 0;
for j = 1:nsd
    for i = 1:nsd
        blk = blk + 1;
        R_all((blk-1)*nzb + 1 : blk*nzb) = r + (i - 1) * N;
        C_all((blk-1)*nzb + 1 : blk*nzb) = cc + (j - 1) * N;
    end
end
switch output
    case 'values'
        t0 = tic; if on_gpu, vals = gather(vals); end; t.download = toc(t0); t.sparse = 0;
        K = vals;
    case 'device_sparse'
        t.download = 0; t0 = tic;
        if on_gpu
            K = sparse(gpuArray(R_all), gpuArray(C_all), double(vals), nsd * N, nsd * N);
        else
            K = sparse(R_all, C_all, double(vals), nsd * N, nsd * N);
        end
        if symmetrize, K = 0.5 * (K + K.'); end
        sync(); t.sparse = toc(t0);
    otherwise % 'cpu_sparse'
        t0 = tic; if on_gpu, vals = gather(vals); end; t.download = toc(t0);
        t0 = tic;
        K = sparse(R_all, C_all, double(vals), nsd * N, nsd * N);
        if symmetrize, K = 0.5 * (K + K.'); end
        t.sparse = toc(t0);
end
t.total = t.upload + t.kernel + t.extract + t.download + t.sparse;
end

function R = contract(sl, WBe, tp, nb, nq, N, nsd)
% Sum factorization of the local SIMP grid with the per-row 1D (weight x basis) matrices.
if nsd == 2
    R = pagemtimes(sl, 'none', WBe{2, tp(2)}, 'transpose');          % [nq1 x nn2 x N]
    R = pagemtimes(WBe{1, tp(1)}, R);                                % [nn1 x nn2 x N]
else
    X = reshape(sl, nq(1), nq(2) * nq(3), N);
    X = reshape(pagemtimes(WBe{1, tp(1)}, X), nb(1), nq(2), nq(3), N);       % dir 1
    X = reshape(permute(X, [2 1 3 4]), nq(2), nb(1) * nq(3), N);
    X = reshape(pagemtimes(WBe{2, tp(2)}, X), nb(2), nb(1), nq(3), N);       % dir 2
    X = reshape(permute(X, [2 1 3 4]), nb(1) * nb(2), nq(3), N);
    R = reshape(pagemtimes(X, 'none', WBe{3, tp(3)}, 'transpose'), nb(1), nb(2), nb(3), N); % dir 3
end
end

function Y = mode_product(A, X, d, nsd)
% Mode-d product of an nsd-dimensional array X with matrix A.
sz = size(X, 1:nsd);
perm = [d, setdiff(1:nsd, d)];
Xd = reshape(permute(X, perm), sz(d), []);
so = sz; so(d) = size(A, 1);
Y = ipermute(reshape(A * Xd, so(perm)), perm);
end
