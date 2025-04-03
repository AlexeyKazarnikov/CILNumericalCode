% This scripts initialises all necessary things for DE optimisation or MCMC sampling.

addpath ../../Startup
nlInitLibrary();

% addpath Please add here the path to the local version of mcmcstat toolbox
% (available from https://mjlaine.github.io/mcmcstat/)

% CIL settings
cil.Nr = 12; % number of radii values
cil.UseUniformBins = true; % use uniform or exponential decay rate
cil.N = 800; % number of simulated patterns in one subset
cil.Ndata = 1; % number of data patterns in one subset
cil.Ntr = 1000; % number of bootstrapped vectors
cil.Ntrial = 100; % number of likelihood evaluations (NEW)

Ndim = 64;
Lmul = 2;

Nsteps = 100;
Npop = 39;

generatorModel = CDIMAModel(round(Lmul * Ndim));
generatorModel.Observer = IdentityObserver();
generatorModel.Settings.ExperimentalSetup = false;

generatorModel.Settings.UseROCK2 = true;
generatorModel.Settings.final_time_point = 200000;

distanceModel = CDIMAModel(Ndim);
distanceModel.Settings.UseROCK2 = true;
distanceModel.DistanceProvider = MultiDistanceProvider(normPower);
distanceModel.Observer = IdentityObserver();
distanceModel.Settings.ExperimentalSetup = true;
distanceModel.DistanceProvider.Settings.UseLinf = false;
distanceModel.DistanceProvider.Settings.UseW1inf = false;
distanceModel.DistanceProvider.Settings.UseW1infPrime = false;

% if you have multiple CUDA devices on your system, you can use them all by
% changing the respective property of distanceModel object. By default the
% first device in the system is used.
% distanceModel.Devices = [...];

grid = UniformGrid(round(Lmul * [Ndim Ndim]));

Sexp = cima_pattern_data.image_data;
Nexp = sqrt(length(Sexp));
gridExp = UniformGrid([Nexp Nexp]);
[Sdata, gridData] = pdResize2(Sexp, gridExp, Ndim);

genMin = [30 10 0.2 100]';
genMax = [41 15 0.6 250]';

popGenerator = @() ...
    genMin + (genMax - genMin) .* rand(length(genMin), Npop);

vec2par = @(theta) struct( ...
    'L', Lmul * theta(1), ...
    'a', theta(2), ...
    'b', theta(3), ...
    'sigma', theta(4), ...
    'd', 1.07 ...
    );


valObjFun = @(theta) ...
    normConst * valObjFun( ...
    theta, ...
    generatorModel, ...
    distanceModel, ...
    Sdata, ...
    grid, ...
    Ndim, ...
    cil, ...
    vec2par);
