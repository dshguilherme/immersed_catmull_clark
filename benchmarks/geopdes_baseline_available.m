function ok = geopdes_baseline_available()
% GEOPDES_BASELINE_AVAILABLE  Optionally put GeoPDEs on the path for baseline comparisons.
%   ok = geopdes_baseline_available() returns true if GeoPDEs (with the NURBS
%   toolbox) can be used. The code base itself does not depend on GeoPDEs; a
%   few benchmarks time it as an external reference implementation. Point the
%   environment variable GEOPDES_PATH at the folder that contains GeoPDEs and
%   nurbs (e.g. ...\geopdes-master); the function adds it to the path once.

ok = exist('op_su_ev', 'file') == 2 && exist('nrbextrude', 'file') == 2;
if ok, return; end
root = getenv('GEOPDES_PATH');
if isempty(root) || ~isfolder(root)
    return;
end
addpath(genpath(root));
ok = exist('op_su_ev', 'file') == 2 && exist('nrbextrude', 'file') == 2;
end
