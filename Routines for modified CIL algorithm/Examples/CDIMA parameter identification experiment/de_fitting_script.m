% This script implements the DE optimisation of the mixed mode CIL cost
% function used for fitting the parameters of the Lengyel-Epstein
% reaction-diffusion model to the experimental chemical data produced by
% CIMA chemical reaction.

clc
clear all
close all


deDataFileName = 'output_de_data.mat'; % file name, to store output data

% loading experimental pattern
cima_pattern_data = load('spot_large_pattern_data.mat'); % spots (hexagons)
% cima_pattern_data = load('stripe_large_pattern_data.mat'); % stripes

% these constants is needed inside init_script.m
normConst = (1 / 50); % spots (hexagons)
% normConst = (1 / 5) * (1 / 50); % stripes
normPower = 2; % spots (hexagons)
% normPower = 5; % stripes

run init_script.m

figure(1); % plotting the experimental pattern
clf;
% minus is added to revert the colours for visualisation
pdPlot2(-Sdata, gridData); 
title('Experimental pattern');

% configuring the DE optimiser
settings = struct();
settings.recalculation_interval = 1;
settings.constraint_fun = @(y) abs(y);

% initialising the logging logic (can be helpful for debugging purposes)
logFileName = 'log_';

dateTimeString = string(datetime);
dateTimeString = strrep(dateTimeString, ' ', '-');
dateTimeString = strrep(dateTimeString, ':', '-');

logFileName = strcat(logFileName, dateTimeString, '.txt');
Logger.logFileName(logFileName);

Logger.log('NEW EXPERIMENT')

% running the DE optimisation algorithm
runDE( ...
    deDataFileName, ...
    popGenerator, ...
    valObjFun, ...
    Nsteps, ...
    settings, ...
    @(data) Logger.log(data) ...
    );

disp('All done!')