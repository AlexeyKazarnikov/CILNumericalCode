% This script implements the uncertainty quantification stage using MCMC 
% sampling.

clc
clear all
close all

% loading experimental pattern
cima_pattern_data = load('spot_large_pattern_data.mat'); % spots (hexagons)
% cima_pattern_data = load('stripe_large_pattern_data.mat'); % stripes

Nchain = 25000;
Nbatch = 100;
mcmcDataFileName = 'mcmc_output.mat'; % file to store the output chain

% these constants is needed inside init_script.m
normConst = (1 / 50); % spots (hexagons)
% normConst = (1 / 5) * (1 / 50); % stripes
normPower = 2; % spots (hexagons)
% normPower = 5; % stripes

run init_script.m

generatorModel.Settings.final_time_point = 200000;

% if needed, DE data can be loaded here and used for creating a starting
% point for MCMC chain
% deDataFileName = 'stripe_de_data_64.mat';
% load(deDataFileName, 'deData');

% output from the DE stage is used as a starting point for MCMC chain
parBest = [36.67 14.39 0.45 45.83]; % spots (hexagons)
%parBest = [38.54 12.61 0.38 122.52]; % stripes

mcmcParams = { ...
    {'L', parBest(1), 0}
    {'a', parBest(2), 0}
    {'b', parBest(3), 0}
    {'sigma', parBest(4), 0}
};

logFileName = 'log_';

dateTimeString = string(datetime);
dateTimeString = strrep(dateTimeString, ' ', '-');
dateTimeString = strrep(dateTimeString, ':', '-');

logFileName = strcat(logFileName, dateTimeString, '.txt');
Logger.logFileName(logFileName);

Logger.log('NEW MCMC EXPERIMENT');

disp('Cost function performance:')
tic;
res = valObjFun(parBest);
toc;

fprintf('Cost function value: %f \n', res);


runMCMC( ...
    valObjFun, ...
    mcmcParams, ...
    mcmcDataFileName, ...
    Nchain, ...
    Nbatch, ...
    [], ...
    1, ...
    {'chainpanel', 'pairs'} ...
    );

disp('All done!')