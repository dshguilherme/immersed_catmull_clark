function knots = iga_open_knots(nel, degree, regularity)
% IGA_OPEN_KNOTS  Uniform open knot vector on [0, 1].
%   knots = iga_open_knots(nel, degree, regularity) builds a knot vector with
%   nel elements, end knots repeated degree+1 times and interior knots repeated
%   degree-regularity times (regularity = degree-1 gives C^{p-1} splines).

if nargin < 3, regularity = degree - 1; end
assert(regularity >= -1 && regularity <= degree - 1, ...
    'iga_open_knots: regularity must lie in [-1, degree-1].');
breaks = linspace(0, 1, nel + 1);
mult = degree - regularity;
knots = [zeros(1, degree + 1), repelem(breaks(2:end-1), mult), ones(1, degree + 1)];
end
