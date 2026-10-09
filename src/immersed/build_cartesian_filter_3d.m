function [H_filter, H_sum] = build_cartesian_filter_3d(grid_res, h_cell, rmin, active_mask)
% BUILD_CARTESIAN_FILTER_3D
% Fast O(N * r^3) convolution filter matrix for 3D topology optimization
% on regular Cartesian background grids.
%
% Inputs:
%   grid_res    - [nx, ny, nz]
%   h_cell      - [hx, hy, hz] element physical dimensions
%   rmin        - Filter radius in physical units (or scalar multiple of mean(h))
%   active_mask - (Optional) logical vector [nel x 1] of design domain elements
%
% Outputs:
%   H_filter    - Sparse filter matrix [nel x nel]
%   H_sum       - Column sum normalization vector [nel x 1]

nx = grid_res(1); ny = grid_res(2); nz = grid_res(3);
nel = nx * ny * nz;

if nargin < 4 || isempty(active_mask)
    active_mask = true(nel, 1);
end

hx = h_cell(1); hy = h_cell(2); hz = h_cell(3);

% Window size in integer offsets
rx_max = ceil(rmin / hx);
ry_max = ceil(rmin / hy);
rz_max = ceil(rmin / hz);

[dI, dJ, dK] = ndgrid(-rx_max:rx_max, -ry_max:ry_max, -rz_max:rz_max);
dI = dI(:); dJ = dJ(:); dK = dK(:);
dist = sqrt((dI * hx).^2 + (dJ * hy).^2 + (dK * hz).^2);
valid_offsets = find(dist <= rmin);

dI = dI(valid_offsets);
dJ = dJ(valid_offsets);
dK = dK(valid_offsets);
dist_weights = rmin - dist(valid_offsets);

num_offsets = numel(valid_offsets);

% Build sparse triplets for active elements
active_indices = find(active_mask);
n_active = numel(active_indices);

i_list = zeros(n_active * num_offsets, 1, 'int32');
j_list = zeros(n_active * num_offsets, 1, 'int32');
s_list = zeros(n_active * num_offsets, 1, 'single');

count = 0;
[ix_act, iy_act, iz_act] = ind2sub([nx, ny, nz], active_indices);

for o = 1:num_offsets
    di = dI(o); dj = dJ(o); dk = dK(o);
    w = dist_weights(o);
    
    % Neighbor indices
    ni = ix_act + di;
    nj = iy_act + dj;
    nk = iz_act + dk;
    
    % Check grid boundaries
    in_bounds = (ni >= 1 & ni <= nx & nj >= 1 & nj <= ny & nk >= 1 & nk <= nz);
    if ~any(in_bounds), continue; end
    
    source_elems = active_indices(in_bounds);
    neigh_elems = sub2ind([nx, ny, nz], ni(in_bounds), nj(in_bounds), nk(in_bounds));
    
    % Only connect if neighbor is also active
    neigh_active = active_mask(neigh_elems);
    if ~any(neigh_active), continue; end
    
    idx_connect = find(neigh_active);
    src = source_elems(idx_connect);
    tgt = neigh_elems(idx_connect);
    n_con = numel(src);
    
    i_list(count+1 : count+n_con) = int32(src);
    j_list(count+1 : count+n_con) = int32(tgt);
    s_list(count+1 : count+n_con) = single(w);
    count = count + n_con;
end

i_list = double(i_list(1:count));
j_list = double(j_list(1:count));
s_list = double(s_list(1:count));

H_filter = sparse(i_list, j_list, s_list, nel, nel);
H_sum = full(sum(H_filter, 2));
H_sum(H_sum < 1e-6) = 1.0;

end
