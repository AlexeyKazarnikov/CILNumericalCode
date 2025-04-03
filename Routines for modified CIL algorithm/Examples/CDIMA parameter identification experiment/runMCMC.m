function [] = runMCMC( ...
    objFun, ...
    mcmcParams, ...
    mcmcFileName, ...
    Nchain, ...
    nBatch, ...
    initCov, ...
    showWaitbar, ...
    plot_options ...
    )
%runMCMC runs the Markov Chain Monte Carlo (MCMC) simulation.
%INPUT
%objFun : Objective function (negative log-likelihood).
%mcmcParams : Parameters for the MCMC simulation.
%mcmcFileName : Name of the file to save or load the MCMC data.
%Nchain : Number of MCMC iterations to perform.
%nBatch : (Optional) Number of samples per batch. Defaults to 100.
%initCov : (Optional) Initial covariance matrix for the MCMC proposal 
% distribution.
%showWaitbar : (Optional) Flag to show a waitbar during the MCMC run. 
% Defaults to true.
%plot_options : (Optional) Cell array of plot options for visualizing 
% the MCMC results.
%OUTPUT
%None (This function does not return any output parameters).


if nargin < 5 || isempty(nBatch)
    nBatch = 100;
end

MCMCModel.ssfun = @(theta,data) objFun(theta);
MCMCModel.sigma2 = 1;
MCMCModel.N = nBatch;

MCMCOptions.nsimu = nBatch;
MCMCOptions.adaptint = nBatch - 1;
MCMCOptions.verbosity = 1;
MCMCOptions.method = 'am';

if nargin < 6 || isempty(initCov)
    Nparams = length(mcmcParams);
    MCMCOptions.qcov = 1e-5 * eye(Nparams, Nparams);
end

if nargin < 7 || isempty(showWaitbar)
    MCMCOptions.waitbar = 1;
else
    MCMCOptions.waitbar = showWaitbar;
end

% mcmc run
if ~exist(mcmcFileName, 'file')
    MCMCRes = [];
    Chain = [];
    SumOfSquares = [];
    Ncomp = 0;
else
    load(mcmcFileName, 'Chain', 'SumOfSquares', 'MCMCRes');
    Ncomp = length(Chain);
end

while Ncomp < Nchain
    disp('Beginning new iteration...')
    [MCMCRes, chain, ~, ss] = ...
        mcmcrun( ...
        MCMCModel, ...
        struct(), ...
        mcmcParams, ...
        MCMCOptions, ...
        MCMCRes ...
        );

    Chain = [Chain; chain];
    SumOfSquares = [SumOfSquares; ss];
    
    disp('Iteration completed!')
    save(mcmcFileName, 'Chain', 'SumOfSquares', 'MCMCRes');

    if nargin > 7 && ~isempty(plot_options)
        for iPlot = 1 : length(plot_options)
            figure(iPlot);
            clf;
            mcmcplot(Chain, [], MCMCRes, plot_options{iPlot});
        end
    end

    Ncomp = Ncomp + MCMCOptions.nsimu;
end

end