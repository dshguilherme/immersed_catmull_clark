function S = wq_setup(space, YOUNG, POISSON, device, precision)
% WQ_SETUP One-time (geometry-dependent) setup for batched weighted-quadrature
% stiffness formation in 2D. Everything computed here is independent of the
% design density and can be reused across all topology optimization iterations.
%
%   space     - displacement space on a 2D box (see IGA_SPACE_BOX)
%   device    - 'cpu' or 'gpu'  (where the batched kernel will run)
%   precision - 'double' (default) or 'single'
%
% The per-iteration work (SIMP scaling + row-wise sum factorization) is done
% by WQ_FORM using the arrays stored in S.

if nargin < 4 || isempty(device), device = 'gpu'; end
if nargin < 5 || isempty(precision), precision = 'double'; end
assert(space.dim == 2, 'wq_setup: only 2D is implemented.');

t_setup = tic;
nsd = 2;
n = space.ndof_dir;
N = prod(n);

%% 1. Univariate WQ rules and basis evaluations (identical to fast_stiffness_assembly)
for d = 1:nsd
    QR(d) = iga_wq_rules_1d(space.knots{d}, space.degree(d)); %#ok<AGROW>
    nb{d} = QR(d).neighbors;
    qn{d} = QR(d).all_points';
end
nqtot = cellfun(@numel, qn);

%% 2. Padded (basis x weight) arrays per direction and per weight type
% Weight types: 1 = W00*B, 2 = W10*B, 3 = W01*B', 4 = W11*B'
for d = 1:nsd
    nd = n(d);
    nnmax(d) = max(cellfun(@numel, nb{d}));
    nqmax(d) = max(QR(d).nquad_points);
    P{d}   = ones(nqmax(d), nd);
    J{d}   = ones(nnmax(d), nd);
    len{d} = zeros(1, nd);
    WB{d}  = zeros(nnmax(d), nqmax(d), nd, 4);
    for ii = 1:nd
        pts = QR(d).ind_points{ii};
        ng  = nb{d}{ii};
        nq = numel(pts); nn = numel(ng);
        P{d}(1:nq, ii) = pts;
        J{d}(1:nn, ii) = ng;
        len{d}(ii) = nn;
        Bv = QR(d).B(pts, ng).';
        Bd = QR(d).dB(pts, ng).';
        WB{d}(1:nn, 1:nq, ii, 1) = QR(d).quad_weights_00{ii} .* Bv;
        WB{d}(1:nn, 1:nq, ii, 2) = QR(d).quad_weights_10{ii} .* Bv;
        WB{d}(1:nn, 1:nq, ii, 3) = QR(d).quad_weights_01{ii} .* Bd;
        WB{d}(1:nn, 1:nq, ii, 4) = QR(d).quad_weights_11{ii} .* Bd;
    end
end

% Row (test function) multi-indices in GeoPDEs ordering (direction 1 fastest)
[I1, I2] = ind2sub(n, 1:N);

%% 3. Metric / constitutive tensor C0 at all WQ points (density independent)
lambda = YOUNG * POISSON / ((1 + POISSON) * (1 - 2 * POISSON));
mu = YOUNG / (2 * (1 + POISSON));
C = iga_wq_elasticity_tensor(space, lambda, mu, nqtot);

% Gather index: for row n, local grid P1(:,I1(n)) x P2(:,I2(n)) -> [nq1 x nq2 x N]
LIN = reshape(P{1}(:, I1), nqmax(1), 1, N) + ...
      (reshape(P{2}(:, I2), 1, nqmax(2), N) - 1) * nqtot(1);

% Pre-gathered C0 pages, stacked over (i,j) along dim 4: C0p{k1,k2} is [nq1 x nq2 x N x 4]
C0p = cell(nsd, nsd);
for k1 = 1:nsd
    for k2 = 1:nsd
        blk = zeros(nqmax(1), nqmax(2), N, 4);
        c = 0;
        for j = 1:nsd
            for i = 1:nsd
                c = c + 1;            % order (1,1),(2,1),(1,2),(2,2)
                Cij = C{i, j}(:, :, k1, k2);
                blk(:, :, :, c) = Cij(LIN);
            end
        end
        C0p{k1, k2} = blk;
    end
end

% Expanded per-row basis x weight arrays: WBe{d,t} is [nn_d x nq_d x N]
WBe = cell(nsd, 4);
for t = 1:4
    WBe{1, t} = WB{1}(:, :, I1, t);
    WBe{2, t} = WB{2}(:, :, I2, t);
end

% Weight type for direction ll given derivative pair (k1,k2) (same rule as CPU loop)
typ = zeros(nsd, nsd, nsd);
for k1 = 1:nsd
    for k2 = 1:nsd
        for ll = 1:nsd
            if k1 == k2 && ll == k1
                typ(k1, k2, ll) = 4;
            elseif k1 ~= k2 && ll == k1
                typ(k1, k2, ll) = 2;
            elseif k1 ~= k2 && ll == k2
                typ(k1, k2, ll) = 3;
            else
                typ(k1, k2, ll) = 1;
            end
        end
    end
end

%% 4. Sparsity pattern (rows/cols) and extraction mask, same ordering as CPU loop
mask = (reshape((1:nnmax(1))', nnmax(1), 1, 1) <= reshape(len{1}(I1), 1, 1, N)) & ...
       (reshape(1:nnmax(2), 1, nnmax(2), 1) <= reshape(len{2}(I2), 1, 1, N));
ROWp = repmat(reshape(1:N, 1, 1, N), nnmax(1), nnmax(2), 1);
COLp = reshape(J{1}(:, I1), nnmax(1), 1, N) + (reshape(J{2}(:, I2), 1, nnmax(2), N) - 1) * n(1);
rows = ROWp(mask);
cols = COLp(mask);

%% 5. Density-to-WQ-point maps
e1 = discretize(qn{1}, space.breaks{1});
e2 = discretize(qn{2}, space.breaks{2});
S1 = QR(1).B;  % [nq1tot x ncp1]
S2 = QR(2).B;  % [nq2tot x ncp2]

%% 6. Move density-independent arrays to the target device / precision
cast_fn = @(x) cast(x, precision);
if strcmpi(device, 'gpu')
    mv = @(x) gpuArray(cast_fn(x));
else
    mv = cast_fn;
end
for t = 1:4
    WBe{1, t} = mv(WBe{1, t});
    WBe{2, t} = mv(WBe{2, t});
end
for k1 = 1:nsd
    for k2 = 1:nsd
        C0p{k1, k2} = mv(C0p{k1, k2});
    end
end
LINd = LIN;
if strcmpi(device, 'gpu'), LINd = gpuArray(uint32(LIN)); else, LINd = uint32(LIN); end
maskd = mask;
if strcmpi(device, 'gpu'), maskd = gpuArray(mask); end
if strcmpi(device, 'gpu')
    e1d = gpuArray(e1); e2d = gpuArray(e2);
    S1d = mv(S1); S2d = mv(S2);
else
    e1d = e1; e2d = e2; S1d = cast_fn(S1); S2d = cast_fn(S2);
end
if strcmpi(device, 'gpu'), wait(gpuDevice); end

S = struct();
S.device = lower(device);
S.precision = precision;
S.N = N; S.n = n; S.nqtot = nqtot; S.nnmax = nnmax; S.nqmax = nqmax;
S.WBe = WBe; S.C0p = C0p; S.typ = typ;
S.LIN = LINd; S.mask = maskd;
S.rows = rows; S.cols = cols;
S.e1 = e1d; S.e2 = e2d; S.S1 = S1d; S.S2 = S2d;
S.nnz_block = numel(rows);
S.t_setup = toc(t_setup);
end
