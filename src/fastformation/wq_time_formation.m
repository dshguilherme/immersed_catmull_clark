function T = wq_time_formation(sp, xPhys, devices, reps)
% WQ_TIME_FORMATION  Measured wall-clock times of weighted-quadrature stiffness
% formation (WQ_SETUP + WQ_FORM) for one density field, for timing benchmarks.
%   T = wq_time_formation(sp, xPhys, devices, reps)
%   devices - cell array of any of 'cpu', 'gpu_fp64', 'gpu_fp32' (default {'cpu'})
%   reps    - repetitions per measurement; the median is reported (default 3)
%   T.<dev>.setup  one-time setup (rules, gather indices, device upload) [s]
%   T.<dev>.form   per-iteration formation of the full sparse K on the host
%                  (SIMP + sum factorization + extraction + transfer + sparse) [s]
%   T.<dev>.kernel per-iteration on-device part only (upload + SIMP + contractions
%                  + extraction, values left on the device) [s]
%   T.<dev>.relerr relative Frobenius difference of K to the CPU FP64 result
% Every number is a measurement; GPU timings synchronize the device.

if nargin < 3 || isempty(devices), devices = {'cpu'}; end
if nargin < 4 || isempty(reps), reps = 3; end
penal = 3; Emin = 1e-3;
Kref = [];
for k = 1:numel(devices)
    dev = devices{k};
    switch dev
        case 'cpu', d = 'cpu'; prec = 'double';
        case 'gpu_fp64', d = 'gpu'; prec = 'double';
        case 'gpu_fp32', d = 'gpu'; prec = 'single';
        otherwise, error('wq_time_formation: unknown device %s', dev);
    end
    if strcmp(d, 'gpu'), reset(gpuDevice); end
    t0 = tic; S = wq_setup(sp, 1.0, 0.3, d, prec); t_setup = toc(t0);
    K = wq_form(S, xPhys, 'element', penal, Emin, true, 'cpu_sparse');    % warm-up
    tf = zeros(reps, 1); tk = zeros(reps, 1);
    for r = 1:reps
        t0 = tic; K = wq_form(S, xPhys, 'element', penal, Emin, true, 'cpu_sparse'); tf(r) = toc(t0);
        [~, t] = wq_form(S, xPhys, 'element', penal, Emin, true, 'values');
        tk(r) = t.upload + t.kernel + t.extract;
    end
    if isempty(Kref), Kref = K; end
    T.(dev) = struct('setup', t_setup, 'form', median(tf), 'kernel', median(tk), ...
        'relerr', norm(K - Kref, 'fro') / norm(Kref, 'fro'), 'nnz', nnz(K));
    clear S K
end
end
