% probe_env.m
fprintf('MATLAB %s\n', version);
g = gpuDevice; fprintf('GPU: %s, CC %s, %.2f GB, driver %s\n', g.Name, g.ComputeCapability, g.TotalMemory/1e9, g.DriverVersion);
try, h = half(1.5); fprintf('half() available on CPU: class %s\n', class(h)); catch ME, fprintf('half() CPU: %s\n', ME.message); end
try, hg = gpuArray(half([1 2 3])); y = hg .* hg; fprintf('half gpuArray: OK (%s)\n', classUnderlying(y)); catch ME, fprintf('half gpuArray: %s\n', ME.message); end
try, A = gpuArray(single(sprand(100,100,0.05))); fprintf('single sparse gpuArray: OK\n'); catch ME, fprintf('single sparse gpuArray: %s\n', ME.message); end
try, A = gpuArray(sprand(100,100,0.05)); x = gpuArray.rand(100,5); y = A*x; fprintf('double sparse gpuArray*dense: OK\n'); catch ME, fprintf('double sparse: %s\n', ME.message); end
try, A = gpuArray(sprand(100,100,0.05)); x = gpuArray(single(rand(100,5))); y = A*x; fprintf('double sparse * single dense: class %s\n', classUnderlying(y)); catch ME, fprintf('mixed sparse: %s\n', ME.message); end
try, [s,~]=system('nvcc --version'); fprintf('nvcc status %d\n', s); catch, end
try, cc = mex.getCompilerConfigurations('C++','Selected'); if isempty(cc), fprintf('No C++ compiler\n'); else, fprintf('C++ compiler: %s\n', cc.Name); end, catch ME, fprintf('compiler: %s\n', ME.message); end
fprintf('pagemtimes gpu single: '); try, a=gpuArray.rand(9,9,10,'single'); b=pagemtimes(a,a); fprintf('OK\n'); catch ME, fprintf('%s\n', ME.message); end
