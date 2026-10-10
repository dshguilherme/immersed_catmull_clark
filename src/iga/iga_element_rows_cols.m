function [rows, cols] = iga_element_rows_cols(conn)
% IGA_ELEMENT_ROWS_COLS  Sparse-assembly indices for element matrices.
%   [rows, cols] = iga_element_rows_cols(conn) with conn [nsh x nel] returns
%   column vectors of length nsh^2 * nel, matching the column-major layout of
%   Ke [nsh x nsh x nel] (rows = test, cols = trial), so that
%   K = sparse(rows, cols, Ke(:), ndof, ndof).

[nsh, nel] = size(conn);
rows = reshape(repmat(reshape(conn, nsh, 1, nel), 1, nsh, 1), [], 1);
cols = reshape(repmat(reshape(conn, 1, nsh, nel), nsh, 1, 1), [], 1);
end
