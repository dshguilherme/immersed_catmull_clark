function f = iga_cantilever_tip_load(x, y, L, h, position)
% IGA_CANTILEVER_TIP_LOAD  Distributed downward load near the free end of a 2D cantilever.
%   f = iga_cantilever_tip_load(x, y, L, h, position) returns [2 x size(x)] with
%   f_y = -1 on the patch x >= L - d and, for position = 'center' (default),
%   |y - h/2| <= d/2, or for 'bottom', y <= d, where d = h/10. Use with
%   IGA_LOAD_VECTOR; callers usually normalise the resulting force vector.
%
%   This replaces forceCantileverCentered / forceCantileverBottom from the
%   external topopt helpers, which hard-coded L = 1, h = 0.5 and, through
%   f(2, i, j) = -1 with vector subscripts, also loaded points outside the patch.

if nargin < 3 || isempty(L), L = 1; end
if nargin < 4 || isempty(h), h = 0.5; end
if nargin < 5 || isempty(position), position = 'center'; end
d = h / 10;
switch lower(position)
    case 'center'
        in_patch = (x >= L - d) & (y <= h/2 + d/2) & (y >= h/2 - d/2);
    case 'bottom'
        in_patch = (x >= L - d) & (y <= d);
    otherwise
        error('iga_cantilever_tip_load: unknown position ''%s''.', position);
end
f = zeros([2, size(x)]);
f(2, :, :) = reshape(-double(in_patch), [1, size(x)]);
end
