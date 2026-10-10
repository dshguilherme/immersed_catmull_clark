function Q = iga_wq_rules_1d(knots, degree)
% IGA_WQ_RULES_1D  Univariate weighted-quadrature rules for stiffness formation.
%   Q = iga_wq_rules_1d(knots, degree) implements the row-wise weighted
%   quadrature of Calabro, Sangalli & Tani, CMAME 316 (2017), for a uniform
%   open knot vector. For each test function B_i it returns weights on the
%   quadrature points inside supp(B_i) such that
%     W00: sum_q w_q g(x_q) = int B_i  g   for g in S_p
%     W10: sum_q w_q g(x_q) = int B_i' g   for g in S_p
%     W01: sum_q w_q g(x_q) = int B_i  g   for g in S_{p-1}  (derivative space)
%     W11: sum_q w_q g(x_q) = int B_i' g   for g in S_{p-1}
%   (minimum-norm solution of the exactness conditions). Reference integrals use
%   (p+1)-point Gauss per element, which is exact for these polynomial products.
%
%   Fields:
%     all_points    [1 x nq]   WQ points: p+2 uniform points in the end elements,
%                              knots and midpoints in the interior elements
%     quad_points{i}, ind_points{i}, nquad_points(i)  points with B_i(x) ~= 0
%     neighbors{i}  S_p functions whose support overlaps supp(B_i)
%     quad_weights_00/_10/_01/_11 {i}  [1 x nquad_points(i)]
%     B, dB         [nq x ndof] values / parametric derivatives of S_p at all_points

knots = knots(:).';
p = degree;
assert(p >= 1, 'iga_wq_rules_1d: degree must be >= 1.');
brk = unique(knots);
nel = numel(brk) - 1;
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
Q.quad_weights_00 = cell(ndof, 1);
Q.quad_weights_10 = cell(ndof, 1);
Q.quad_weights_01 = cell(ndof, 1);
Q.quad_weights_11 = cell(ndof, 1);

for i = 1:ndof
    ind = find(B(:, i) ~= 0).';
    Q.ind_points{i} = ind;
    Q.quad_points{i} = all_points(ind);
    Q.nquad_points(i) = numel(ind);
    els = sp_p.supp{i};
    Q.neighbors{i} = unique(sp_p.connectivity(:, els)).';
    nb_d = unique(sp_d.connectivity(:, els)).';

    % Gauss data on supp(B_i): test values/derivatives and trial values.
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
