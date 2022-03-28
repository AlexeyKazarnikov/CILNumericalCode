% This example shows how to use the library to create CIL likelihood
% by using a single distance and next construct parameter posterior
% distribution by using MCMC methods.

clc
clear all
close all

% first add path to the library and to the mcmcstat toolbox
addpath D:\Programming\GitHub\NewLibrary\mcmcstat
addpath ../Startup

% this line initializes all library paths, allowing to use all functions
% and classes
nlInitLibrary();

% next we create an instance of Model class, which is an abstraction for
% all models, used here
model = FHNModel(64); % creates an instance of FHNModel
% we can create an instance of any other model in a very similar way, for
% example:
% model = BZModel(64); % creates an instance of Brusselator model

% next line simulates a set of patterns
s = model.simulate(4);

% each model has built-in routine for easy data visualization
figure(1)
clf
subplot(2,2,1)
model.visualize(s,1);
subplot(2,2,2)
model.visualize(s,2);
subplot(2,2,3)
model.visualize(s,3);
subplot(2,2,4)
model.visualize(s,4);

pause

% next we generate the CIL likelihood

disp('Beginning CIL likelihood generation...')

% if needed, the default L2-based distance, used by the model, can be changed to any instance
% of DistanceProvider class.
% for example:
% model.DistanceProvider = LpDistance(0); %L-infinity distance
model.DistanceProvider = W1pDistance(2); % Sobolev space W12 distance

builder = DefaultCILBuilder(model); % creating an instance of CIL generator

theta0 = struct(); % vector of control parameters (if empty, default values will be used)
% default model parameters are located in Parameters field
disp(model.Parameters)

% configuring CIL builder
builder.CIL.N = 50; % number of patterns in one trajectory
builder.CIL.M = 13; % dimension of CIL vector
builder.Settings.UseUniformSpacing = false; % use uniform spacing of bins or power law
builder.Settings.UseBootstrap = true; % enable bootstrapping

% first we estimate eCDF constants
builder.estimate_ecdf_constants(theta0);
[c0,x0] = builder.generate_ecdf(theta0);

figure(2)
clf
plot(x0,c0,'ro-')
title('An example of eCDF vector')

pause

% we continue with generating pattern data for CIL likelihood
builder.generate_pattern_data(theta0);

% based on pattern data we generate the distribution of CIL vectors
builder.generate_distribution(false);

figure(3)
clf
plot(builder.Y','bo-')
title('The distribution of CIL vectors')

pause

% next we perform chi2-squared test and if everything is fine, continuing
% with estimating CIL parameters

figure(4)
clf
builder.perform_chi2_test();
title('Chi2 Normality test')

pause

builder.estimate_distribution_parameters();

% finally we serialize the data and pass it to MCMC sampler
data = builder.serialize();

disp('Beginning MCMC samling...')

% before doing MCMC sampling we specify the sampling parameters
params = { ...
            {'mu',  model.Parameters.mu, 0}, ...
            {'eps',  model.Parameters.eps, 0} ...
         };

% creating the instance of MCMC sampler class
runner = MCMCRunner(data, params);

% BUG-FIX to avoid different precision errors (to be fixed in future)
data.S = single(data.S);

% generating MCMC chain of length 5000
runner.run(5000,data);

% plotting MCMC chain
figs = runner.plot_chain(4 + [1 2 3 4]);

disp('All done!')
