function [elem_weights, status] = assemble_immersed_element_weights(brep, grid_bounds, grid_res, subcell_res)
% ASSEMBLE_IMMERSED_ELEMENT_WEIGHTS
% Computes the effective integration weight w_e in [0, 1] for all background
% Catmull-Clark cells embedding a B-Rep model:
%   - Uncut interior: w_e = 1.0
%   - Uncut exterior: w_e = 0.0 (or small epsilon for numerical stability)
%   - Cut cells: w_e in (0, 1) computed via sub-cell quadrature

if nargin < 4 || isempty(subcell_res), subcell_res = [4, 4, 4]; end

nx = grid_res(1); ny = grid_res(2); nz = grid_res(3);
xmin = grid_bounds(1,1); xmax = grid_bounds(1,2);
ymin = grid_bounds(2,1); ymax = grid_bounds(2,2);
zmin = grid_bounds(3,1); zmax = grid_bounds(3,2);

hx = (xmax - xmin) / nx;
hy = (ymax - ymin) / ny;
hz = (zmax - zmin) / nz;

% 1. Classify cells
status = classify_background_cells(brep, grid_bounds, grid_res);

elem_weights = zeros(nx, ny, nz, 'single');
% Interior cells get weight 1.0
elem_weights(status == 1) = 1.0;

% 2. Process cut cells with sub-cell quadrature
cut_indices = find(status == -1);
fprintf('Evaluating sub-cell quadrature for %d cut cells...\n', numel(cut_indices));

[ix_cut, iy_cut, iz_cut] = ind2sub([nx, ny, nz], cut_indices);

t0 = tic;
for k = 1:numel(cut_indices)
    i = ix_cut(k); j = iy_cut(k); l = iz_cut(k);
    
    c_bounds = [xmin + (i-1)*hx, xmin + i*hx; ...
                ymin + (j-1)*hy, ymin + j*hy; ...
                zmin + (l-1)*hz, zmin + l*hz];
            
    w_cut = compute_cut_cell_quadrature(brep, c_bounds, subcell_res);
    elem_weights(i, j, l) = single(w_cut);
end
fprintf('Cut-cell quadrature complete in %.2f s.\n', toc(t0));

end
