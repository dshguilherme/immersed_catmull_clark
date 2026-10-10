function sp = iga_space_box(bounds, nsub, degree, regularity, nquad)
% IGA_SPACE_BOX  Tensor-product B-spline vector space on an axis-aligned box.
%   sp = iga_space_box(bounds, nsub, degree, regularity, nquad)
%     bounds     [dim x 2]  physical box [min max] per direction (dim = 2 or 3)
%     nsub       [1 x dim]  elements per direction
%     degree     scalar or [1 x dim]
%     regularity scalar or [1 x dim] (default degree-1)
%     nquad      scalar or [1 x dim] Gauss points per direction (default degree+1)
%
%   The geometry map is affine: x_d = bounds(d,1) + L_d * xi_d, xi in [0,1]^dim.
%   Numbering follows the GeoPDEs conventions so results are interchangeable:
%     - elements and scalar control points are lexicographic, direction 1 fastest;
%     - vector DOFs are blocked by component: [u_1(:); u_2(:); u_3(:)];
%     - local element functions are lexicographic, direction 1 fastest, then
%       blocked by component;
%     - boundary sides: 1/2 = xi_1 = 0/1, 3/4 = xi_2 = 0/1, 5/6 = xi_3 = 0/1.
%
%   Main fields: dim, ncomp, bounds, L, h, nel_dir, nel, degree, knots{d},
%   breaks{d}, univ(d) (see IGA_SPACE_1D), ndof_dir, ndof_sc, ndof, nsh_sc,
%   nsh, nsh_max, nqn, conn_sc [nsh_sc x nel], connectivity [nsh x nel],
%   boundary(side).dofs_sc / .dofs.

dim = size(bounds, 1);
assert(any(dim == [2 3]) && size(bounds, 2) == 2, 'iga_space_box: bounds must be [dim x 2], dim = 2 or 3.');
expand = @(v) v(:).' .* ones(1, dim);
degree = expand(degree);
if nargin < 4 || isempty(regularity), regularity = degree - 1; end
if nargin < 5 || isempty(nquad), nquad = degree + 1; end
regularity = expand(regularity);
nquad = expand(nquad);
nsub = nsub(:).';
assert(numel(nsub) == dim, 'iga_space_box: nsub must have one entry per direction.');

sp.dim = dim;
sp.ncomp = dim;
sp.bounds = bounds;
sp.L = (bounds(:, 2) - bounds(:, 1)).';
sp.nel_dir = nsub;
sp.nel = prod(nsub);
sp.h = sp.L ./ nsub;
sp.degree = degree;
sp.regularity = regularity;
sp.nquad = nquad;
sp.nqn = prod(nquad);

for d = 1:dim
    knots = iga_open_knots(nsub(d), degree(d), regularity(d));
    univ(d) = iga_space_1d(knots, degree(d), nquad(d)); %#ok<AGROW>
    sp.knots{d} = univ(d).knots;
    sp.breaks{d} = univ(d).breaks;
end
sp.univ = univ;

sp.ndof_dir = [univ.ndof];
sp.ndof_sc = prod(sp.ndof_dir);
sp.ndof = dim * sp.ndof_sc;
nsh_dir = degree + 1;
sp.nsh_dir = nsh_dir;
sp.nsh_sc = prod(nsh_dir);
sp.nsh = dim * sp.nsh_sc;
sp.nsh_max = sp.nsh;

% Scalar connectivity: tensor product of the 1D connectivities.
sub = cell(1, dim);
[sub{:}] = ind2sub(nsub, 1:sp.nel);
loc = cell(1, dim);
[loc{:}] = ind2sub(nsh_dir, (1:sp.nsh_sc).');
gidx = cell(1, dim);
for d = 1:dim
    c1 = univ(d).connectivity;                  % [(p+1) x nel_d]
    gidx{d} = c1(sub2ind(size(c1), repmat(loc{d}, 1, sp.nel), repmat(sub{d}, sp.nsh_sc, 1)));
end
sp.conn_sc = sub2ind(sp.ndof_dir, gidx{:});    % [nsh_sc x nel]
sp.connectivity = sp.conn_sc + reshape((0:dim-1) * sp.ndof_sc, 1, 1, dim);
sp.connectivity = reshape(permute(sp.connectivity, [1 3 2]), sp.nsh, sp.nel);

% Boundary DOFs per side.
cp = cell(1, dim);
[cp{:}] = ind2sub(sp.ndof_dir, 1:sp.ndof_sc);
for d = 1:dim
    for side = 1:2
        if side == 1, sel = find(cp{d} == 1); else, sel = find(cp{d} == sp.ndof_dir(d)); end
        iside = 2 * (d - 1) + side;
        sp.boundary(iside).dofs_sc = sel;
        sp.boundary(iside).dofs = reshape(sel(:) + (0:dim-1) * sp.ndof_sc, 1, []);
    end
end
end
