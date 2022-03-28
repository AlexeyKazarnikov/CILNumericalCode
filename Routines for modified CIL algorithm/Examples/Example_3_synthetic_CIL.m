% This example demonstrates the application of synthetic CIL likelihood and
% constructs the parameter posterior distribution in the case of one data
% pattern.

clc
clear all
close all

% first add path to the library and to the mcmcstat toolbox
addpath ../Startup
addpath D:\Programming\GitHub\NewLibrary\mcmcstat

% this line initializes all library paths, allowing to use all functions
% and classes
nlInitLibrary();

% next we create an instance of a model and define control parameters for
% sampling
model = BZModel(64);
model.DistanceProvider = MultiDistanceProvider(); % multi-norm usage added
params = {
    {'A',  model.Parameters.A, 0}
    {'B',  model.Parameters.B, 0}
    };

% we create and set up the cost function for sampling
cf = CILSyntheticCostFunction();
cf.Settings.UseUniformBins = true;
cf.Settings.Nr = 12;

Nchain = 1000; % length of parameter chain

% generating synthetic data
sref = model.simulate(1);

figure(1)
clf
model.visualize(sref,1)
xlabel('x_1')
ylabel('x_2')

% creating a structure for further usage by 'MCMCRunner' class
exp_data.S = sref;
exp_data.Model = model.serialize();
exp_data.CostFunction = cf.serialize();
exp_data.Type = 'Synthetic';

% creating an instance of MCMC runner
runner = MCMCRunner(exp_data,params);

tic;
fprintf('Value of the cost function for staring point: %.2f\n',...
        runner.ss_fun([params{1}{2} params{2}{2}],exp_data) ...
        );
toc;

% running MCMC sampling
runner.run(Nchain,exp_data);

% plotting MCMC chain
figs = runner.plot_chain(1 + [1 2 3 4]);

disp('All done!')




