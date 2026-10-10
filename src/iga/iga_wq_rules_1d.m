function Q = iga_wq_rules_1d(knots, degree, layout)
% IGA_WQ_RULES_1D  Univariate weighted-quadrature rules for stiffness formation.
%   Q = iga_wq_rules_1d(knots, degree)            interior layout (default)
%   Q = iga_wq_rules_1d(knots, degree, 'calabro') original layout of Calabro,
%       Sangalli & Tani, CMAME 316 (2017): p+2 uniform points in the end elements,
%       knots and midpoints in the interior elements (points on element interfaces).
%
%   Interior layout (FastFormation paper): points strictly inside the elements,
%       x_{k,m} = x_{k-1} + (2m-1)/(2 q_k) h_k, m = 1..q_k,
%   with q_k = 3, and q_k = max(3, p+1) in the first and last element, so that no
%   point lies on an element interface (element-wise densities are single-valued at
%   every point). For each test function B_i the weights are
%     W0: sum_q w_q g(x_q) = int B_i  g,   W1: sum_q w_q g(x_q) = int B_i' g,
%   exact for every g in T = S_p^{p-2} (degree p, continuity C^{p-2}). T contains
%   both the trial functions S_p and their derivatives, so W0/W1 serve all four
%   stiffness terms and the formed matrix is exact for constant coefficients.
%   W0 is the weighted minimum-norm solution, min ||Z^-1 w|| with
%   Z = diag(B_i(x_q) h_q), and W1 follows from the derivative recurrence
%   B_i' = a_i N_{i,p-1} - b_i N_{i+1,p-1}, with W0-type rules for the two
%   degree-(p-1) functions.
%
%   Fields (both layouts):
%     all_points    [1 x nq]   quadrature points
%     quad_points{i}, ind_points{i}, nquad_points(i)  points in supp(B_i)
%     neighbors{i}  S_p functions whose support overlaps supp(B_i)
%     quad_weights_00/_10/_01/_11 {i}: weights for (test value / test derivative) x
%                   (trial value / trial derivative). In the interior layout
%                   _01 = _00 (= W0) and _11 = _10 (= W1).
%     B, dB         [nq x ndof] values / parametric derivatives of S_p at all_points
%     layout        'interior' or 'calabro'

if nargin < 3 || isempty(layout), layout = 'interior'; end
knots = knots(:).';
p = degree;
assert(p >= 1, 'iga_wq_rules_1d: degree must be >= 1.');
switch lower(layout)
    case 'interior'
        Q = rules_interior(knots, p);
    case 'calabro'
        Q = rules_calabro(knots, p);
    otherwise
        error('iga_wq_rules_1d: unknown layout ''%s''.', layout);
end
Q.layout = lower(layout);
end

function Q = rules_interior(knots, p)
brk = unique(knots);
nel = numel(brk) - 1;
ndof = numel(knots) - p - 1;
h = diff(brk);

% Interior points and the element / length of each point
pts = []; el = [];
for k = 1:nel
    if k == 1 || k == nel
        q = max(3, p + 1);
    else
        q = 3;
    end
    m = 1:q;
    pts = [pts, brk(k) + (2 * m - 1) / (2 * q) * h(k)]; %#ok<AGROW>
    el = [el, k * ones(1, q)]; %#ok<AGROW>
end
Q.all_points = pts;
hq = h(el);
[B, dB] = iga_bspline_basis(knots, p, pts);
Q.B = B; Q.dB = dB;

% Trial space T = S_p^{p-2} (interior knots repeated twice) and degree p-1 test functions
knots_T = [brk(1) * ones(1, p + 1), repelem(brk(2:end-1), 2), brk(end) * ones(1, p + 1)];
BT = iga_bspline_basis(knots_T, p, pts);
sp_p = iga_space_1d(knots, p, p + 1);
sp_T = iga_space_1d(knots_T, p, p + 1);
Bm = iga_bspline_basis(knots, p - 1, pts);          % [nq x (ndof+1)]

Q.nquad_points = zeros(ndof, 1);
Q.quad_points = cell(ndof, 1);
Q.ind_points = cell(ndof, 1);
Q.neighbors = cell(1, ndof);
[Q.quad_weights_00, Q.quad_weights_10, Q.quad_weights_01, Q.quad_weights_11] = deal(cell(ndof, 1));

for i = 1:ndof
    els = sp_p.supp{i};
    ind = find(ismember(el, els));
    Q.ind_points{i} = ind;
    Q.quad_points{i} = pts(ind);
    Q.nquad_points(i) = numel(ind);
    Q.neighbors{i} = unique(sp_p.connectivity(:, els)).';

    % W0: exact for T on supp(B_i), weighted by Z = B_i(x_q) h_q
    w0 = wq_rule(B(ind, i), ind, els, @(xg) iga_bspline_basis(knots, p, xg), i);
    % W1 by recurrence: B_i' = a N_{i,p-1} - b N_{i+1,p-1}
    da = knots(i + p) - knots(i);
    db = knots(i + p + 1) - knots(i + 1);
    w1 = zeros(1, numel(ind));
    if da > 0
        w1 = w1 + (p / da) * wq_rule(Bm(ind, i), ind, els, @(xg) iga_bspline_basis(knots, p - 1, xg), i);
    end
    if db > 0
        w1 = w1 - (p / db) * wq_rule(Bm(ind, i + 1), ind, els, @(xg) iga_bspline_basis(knots, p - 1, xg), i + 1);
    end
    Q.quad_weights_00{i} = w0;  Q.quad_weights_01{i} = w0;
    Q.quad_weights_10{i} = w1;  Q.quad_weights_11{i} = w1;
end

    function w = wq_rule(mvals, ind, els, testfun, j)
        % Weighted min-norm weights on points ind for test function j (values mvals
        % at those points): sum_q w_q t(x_q) = int t * test for all t in T.
        z = (mvals(:) .* hq(ind).').';
        act = z > 0;
        w = zeros(1, numel(ind));
        if ~any(act), return; end
        xg = reshape(sp_p.qn(:, els), [], 1);
        wg = reshape(sp_p.qw(:, els), [], 1);
        Mg = testfun(xg); Mg = Mg(:, j);
        Tg = iga_bspline_basis(knots_T, p, xg);
        nbT = unique(sp_T.connectivity(:, unique(el(ind(act)))));
        nbT = nbT(:).';
        A = BT(ind(act), nbT).';               % [ntrial x nact]
        keep = any(abs(A) > 0, 2);             % trial functions seen by the active points
        A = A(keep, :);
        rhs = Tg(:, nbT(keep)).' * (wg .* Mg);
        % w = Z (A Z)^+ rhs via QR of (A Z)^T (avoids squaring the condition number)
        za = z(act);
        [Qa, Ra] = qr((A .* za).', 0);
        w(act) = za .* (Qa * (Ra.' \ rhs)).';
    end
end

function Q = rules_calabro(knots, p)
brk = unique(knots);
ndof = numel(knots) - p - 1;
first = linspace(brk(1), brk(2), p + 2);
last = linspace(brk(end-1), brk(end), p + 2);
interior = sort([brk(3:end-2), 0.5 * (brk(2:end-2) + brk(3:end-1))]);
all_points = unique([first, interior, last]);
Q.all_points = all_points;
[B, dB] = iga_bspline_basis(knots, p, all_points);
Q.B = B;
Q.dB = dB;
% Spaces: S_p (test and trial) and S_{p-1} (trial for the derivative rules).
sp_p = iga_space_1d(knots, p, p + 1);
knots_d = knots(2:end-1);
sp_d = iga_space_1d(knots_d, p - 1, p + 1);
Bd = iga_bspline_basis(knots_d, p - 1, all_points);
Q.nquad_points = zeros(ndof, 1);
Q.quad_points = cell(ndof, 1);
Q.ind_points = cell(ndof, 1);
Q.neighbors = cell(1, ndof);
[Q.quad_weights_00, Q.quad_weights_10, Q.quad_weights_01, Q.quad_weights_11] = deal(cell(ndof, 1));
for i = 1:ndof
    ind = find(B(:, i) ~= 0).';
    Q.ind_points{i} = ind;
    Q.quad_points{i} = all_points(ind);
    Q.nquad_points(i) = numel(ind);
    els = sp_p.supp{i};
    Q.neighbors{i} = unique(sp_p.connectivity(:, els)).';
    nb_d = unique(sp_d.connectivity(:, els)).';
    xg = reshape(sp_p.qn(:, els), [], 1);
    wg = reshape(sp_p.qw(:, els), [], 1);
    [Bg, dBg] = iga_bspline_basis(knots, p, xg);
    Bdg = iga_bspline_basis(knots_d, p - 1, xg);
    Ap = B(ind, Q.neighbors{i}).';
    Ad = Bd(ind, nb_d).';
    Q.quad_weights_00{i} = solve_min_norm(Ap, Bg(:, Q.neighbors{i}).' * (wg .* Bg(:, i)));
    Q.quad_weights_10{i} = solve_min_norm(Ap, Bg(:, Q.neighbors{i}).' * (wg .* dBg(:, i)));
    Q.quad_weights_01{i} = solve_min_norm(Ad, Bdg(:, nb_d).' * (wg .* Bg(:, i)));
    Q.quad_weights_11{i} = solve_min_norm(Ad, Bdg(:, nb_d).' * (wg .* dBg(:, i)));
end
end

function w = solve_min_norm(A, rhs)
% Minimum-norm solution of A w = rhs (A has full row rank), returned as a row.
[Qa, Ra] = qr(A.', 0);
w = (Qa * (Ra.' \ rhs)).';
end
