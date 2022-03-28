% This example shows how to use the library to create CIL likelihood
% by using a multiple distances and next construct parameter posterior
% distribution by using MCMC methods.

clc
clear all
close all

% first add path to the library and to the mcmcstat toolbox
addpath ../Startup
addpath D:\Programming\GitHub\NewLibrary\mcmcstat

% this line initializes all library paths, allowing to use all functions
% and classes
nlInitLibrary();

% next we create an instance of Model class, which is an abstraction for
% all models, used here
model = GMModel(64); % creates an instance of FHNModel
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

% first we define distances, which we will use in the creation of the CIL
% likelihood.
distance_providers = {};
distance_providers{1} = LpDistance(2);
distance_providers{2} = LpDistance(0);
distance_providers{3} = W1pDistance(2);
distance_providers{4} = W1pDistance(0);

distance_names = {'L_2','L_{\infty}','W_{1,2}','W_{1,\infty}'};

builder = MultiCILBuilder( ...
    model, ...
    distance_providers ...
    ); % creating an instance of CIL generator

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
for k=1:4
    subplot(2,2,k)
    plot(x0{k},c0{k},'ro-')
    xlabel(distance_names{k})
end

pause

% we continue with generating pattern data for CIL likelihood
builder.generate_pattern_data(theta0);

% based on pattern data we generate the distribution of CIL vectors
builder.generate_distribution(false);

figure(3)
clf
for k=1:4
    subplot(2,2,k)
    plot(builder.Y(:,:,k)','bo-')
    xlabel(distance_names{k})
end

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
            {'mua',  model.Parameters.mua, 0}, ...
            {'mui',  model.Parameters.mui, 0} ...
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
