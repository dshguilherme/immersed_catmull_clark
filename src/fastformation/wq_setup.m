function S = wq_setup(space, YOUNG, POISSON, device, precision, layout)
% WQ_SETUP One-time (geometry-dependent) setup for batched weighted-quadrature
% stiffness formation in 2D or 3D. Everything computed here is independent of the
% design density and is reused across all topology optimization iterations.
%
%   space     - displacement space on a box (see IGA_SPACE_BOX), dim = 2 or 3
%   device    - 'cpu' or 'gpu'  (where the batched kernel will run)
%   precision - 'double' (default) or 'single'
%   layout    - WQ point layout, 'interior' (default) or 'calabro' (IGA_WQ_RULES_1D)
%
% On a box the metric-constitutive tensor is constant, C_ijkl(x) = c_ijkl * simp(x),
% so per row and per derivative pair (k1, k2) one sum-factorized contraction of the
% SIMP factor serves all d^2 displacement blocks. The per-iteration work is done by
% WQ_FORM using the arrays stored in S.

if nargin < 4 || isempty(device), device = 'gpu'; end
if nargin < 5 || isempty(precision), precision = 'double'; end
if nargin < 6 || isempty(layout), layout = 'interior'; end
t_setup = tic;
nsd = space.dim;
n = space.ndof_dir;
N = prod(n);

%% 1. Univariate WQ rules and padded (weight x basis) arrays per direction and type
% Weight types: 1 = W00*B, 2 = W10*B, 3 = W01*B', 4 = W11*B'
for d = 1:nsd
    QR(d) = iga_wq_rules_1d(space.knots{d}, space.degree(d), layout); %#ok<AGROW>
    qn{d} = QR(d).all_points'; %#ok<AGROW>
    nd = n(d);
    nnmax(d) = max(cellfun(@numel, QR(d).neighbors)); %#ok<AGROW>
    nqmax(d) = max(QR(d).nquad_points); %#ok<AGROW>
    P{d} = ones(nqmax(d), nd); J{d} = ones(nnmax(d), nd); len{d} = zeros(1, nd); %#ok<AGROW>
    WB{d} = zeros(nnmax(d), nqmax(d), nd, 4); %#ok<AGROW>
    for ii = 1:nd
        pts = QR(d).ind_points{ii}; ng = QR(d).neighbors{ii};
        nq = numel(pts); nn = numel(ng);
        P{d}(1:nq, ii) = pts; J{d}(1:nn, ii) = ng; len{d}(ii) = nn;
        Bv = QR(d).B(pts, ng).'; Bd = QR(d).dB(pts, ng).';
        WB{d}(1:nn, 1:nq, ii, 1) = QR(d).quad_weights_00{ii} .* Bv;
        WB{d}(1:nn, 1:nq, ii, 2) = QR(d).quad_weights_10{ii} .* Bv;
        WB{d}(1:nn, 1:nq, ii, 3) = QR(d).quad_weights_01{ii} .* Bd;
        WB{d}(1:nn, 1:nq, ii, 4) = QR(d).quad_weights_11{ii} .* Bd;
    end
end
nqtot = cellfun(@numel, qn);

% Row (test function) multi-indices, direction 1 fastest
I = cell(1, nsd);
[I{:}] = ind2sub(n, 1:N);

%% 2. Constant coefficients c(i,j,k1,k2) (parametric, Jacobian included)
lambda = YOUNG * POISSON / ((1 + POISSON) * (1 - 2 * POISSON));
mu = YOUNG / (2 * (1 + POISSON));
Cc = iga_wq_elasticity_tensor(space, lambda, mu, ones(1, nsd));
c = zeros(nsd, nsd, nsd, nsd);
for i = 1:nsd
    for j = 1:nsd
        c(i, j, :, :) = reshape(Cc{i, j}, [1 1 nsd nsd]);
    end
end

%% 3. Gather index of each row's local point grid in the global grid: [nq1 x .. x nqd x N]
LIN = zeros([nqmax, N]);
stride = 1;
for d = 1:nsd
    shp = ones(1, nsd + 1); shp(d) = nqmax(d); shp(end) = N;
    LIN = LIN + (reshape(P{d}(:, I{d}), shp) - 1) * stride;
    stride = stride * nqtot(d);
end
LIN = LIN + 1;

% Expanded per-row (weight x basis) arrays: WBe{d,t} is [nn_d x nq_d x N]
WBe = cell(nsd, 4);
for d = 1:nsd
    for t = 1:4
        WBe{d, t} = WB{d}(:, :, I{d}, t);
    end
end

% Weight type for direction ll given derivative pair (k1,k2)
typ = zeros(nsd, nsd, nsd);
for k1 = 1:nsd
    for k2 = 1:nsd
        for ll = 1:nsd
            if k1 == k2 && ll == k1, typ(k1, k2, ll) = 4;
            elseif k1 ~= k2 && ll == k1, typ(k1, k2, ll) = 2;
            elseif k1 ~= k2 && ll == k2, typ(k1, k2, ll) = 3;
            else, typ(k1, k2, ll) = 1;
            end
        end
    end
end

%% 4. Sparsity pattern of one block and the extraction mask
mask = true([nnmax, N]);
COL = zeros([nnmax, N]);
stride = 1;
for d = 1:nsd
    shp = ones(1, nsd + 1); shp(d) = nnmax(d); shp(end) = N;
    lshp = ones(1, nsd + 1); lshp(d) = nnmax(d);
    mask = mask & (reshape((1:nnmax(d)).', lshp) <= reshape(len{d}(I{d}), [ones(1, nsd), N]));
    COL = COL + (reshape(J{d}(:, I{d}), shp) - 1) * stride;
    stride = stride * n(d);
end
COL = COL + 1;
ROW = repmat(reshape(1:N, [ones(1, nsd), N]), [nnmax, 1]);
rows = ROW(mask); cols = COL(mask);

%% 5. Density-to-WQ-point maps (element index per point, spline collocation)
for d = 1:nsd
    ebin{d} = discretize(qn{d}, space.breaks{d}); %#ok<AGROW>
    Sd{d} = full(QR(d).B); %#ok<AGROW>
end

%% 6. Move density-independent arrays to the target device / precision
cast_fn = @(x) cast(x, precision);
if strcmpi(device, 'gpu'), mv = @(x) gpuArray(cast_fn(x)); else, mv = cast_fn; end
for d = 1:nsd
    for t = 1:4, WBe{d, t} = mv(WBe{d, t}); end
    Sd{d} = mv(Sd{d});
end
LIN = uint32(LIN); maskd = mask;
if strcmpi(device, 'gpu')
    LIN = gpuArray(LIN); maskd = gpuArray(mask);
    for d = 1:nsd, ebin{d} = gpuArray(ebin{d}); end
    wait(gpuDevice);
end

S = struct('device', lower(device), 'precision', precision, 'dim', nsd, 'N', N, 'n', n, ...
    'nqtot', nqtot, 'nnmax', nnmax, 'nqmax', nqmax, 'c', c, 'typ', typ, 'LIN', LIN, ...
    'mask', maskd, 'rows', rows, 'cols', cols, 'nnz_block', numel(rows), 'layout', lower(layout));
S.WBe = WBe; S.ebin = ebin; S.Sd = Sd;
S.t_setup = toc(t_setup);
end
