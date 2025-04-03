function [] = runDE( ...
    dataFileName, ...
    initialPopulationGenerator, ...
    obj_fun, ...
    Nsteps, ...
    settings, ...
    logFun ...
    )
%runDE runs the Differential Evolution optimization process.
%INPUT
%dataFileName : Name of the file to save or load the optimization data.
%initialPopulationGenerator : Function handle to generate the initial 
% population.
%obj_fun : Objective function to be minimized.
%Nsteps : Number of optimization steps to perform.
%settings : (Optional) Structure containing various settings for the 
% optimization process.
%logFun : (Optional) Function handle for logging progress.
%OUTPUT
%None (This function does not return any output parameters).


if nargin < 6
    logFun = [];
end

if nargin < 5
    settings = struct();
end

if ~isfield(settings, 'output_level')
    settings.output_level = 2;
end
if ~isfield(settings, 'recalculation_interval')
    settings.recalculation_interval = 0;
end
if ~isfield(settings, 'constraint_fun')
    settings.constraint_fun = [];
end

if settings.output_level > 0
    disp('DE optimisation procedure')
    disp('Checking for data from previous run(s)...')
end

if ~exist(dataFileName, 'file')
    if settings.output_level > 0
        disp('DE data file was not found, starting new experiment...')
    end

    if nargin < 2 || isempty(initialPopulationGenerator)
        error('Initial population data are missing!');
    end
    
    deData.Population = initialPopulationGenerator();
    deData.Objectives = [];
    deData.StepNumber = 1;
else
    load(dataFileName, 'deData');
    if settings.output_level > 0
        disp('DE data was loaded from file...');
    end
end

if ~isempty(deData.Objectives)
    optimizer = DEOptimizer( ...
        deData.Population(:, :, end), ...
        deData.Objectives(end, :) ...
        );
else
    optimizer = DEOptimizer( ...
        deData.Population(:, :, end), ...
        [] ...
        );
end

if settings.recalculation_interval > 0
    optimizer.Settings.ObjectiveRecalculation = true;
else
    optimizer.Settings.ObjectiveRecalculation = false;
end

if settings.output_level > 1
    optimizer.Settings.ShowProgress = true;
else
    optimizer.Settings.ShowProgress = false;
end

if settings.output_level > 0
    disp('Initialisation completed!');

    disp('Beginning optimisation procedure...');
end

for k = deData.StepNumber : Nsteps
    if settings.output_level > 0
        fprintf('Beginning step %i / %i...\n', k, Nsteps);
    end

    if nargin > 5 && ~isempty(logFun)
        logFun(sprintf('DE: BEGIN STEP %i', k));
    end
    
    optimizer.minimize(obj_fun, settings.constraint_fun, logFun);

    if nargin > 5 && ~isempty(logFun)
        logFun(sprintf('DE: END STEP %i', k));
    end
    
    if settings.output_level > 0
        fprintf('Step %i / %i completed! \n', k, Nsteps);
    end
    
    if settings.output_level > 0
        [best_candidate, best_value, best_ind] = optimizer.best_candidate();

        if nargin > 5 && ~isempty(logFun)
            logFun(sprintf('DE: BEST CANDIDATE: %i', best_ind));
        end

        disp('Best candidate: ')
        disp(best_candidate);
        disp('Best value: ')
        disp(best_value);
        
        disp('Mean:');
        disp(optimizer.population_mean());
        disp('Std:');
        disp(optimizer.population_std());
        
        disp('Saving data...')
    end

    deData.Population(:, :, end + 1) = optimizer.population();
    deData.Objectives = [deData.Objectives; optimizer.objectives()];
    deData.StepNumber = k + 1;

    save(dataFileName, 'deData');

    if settings.output_level > 0
        disp('Data saved!');
    end
end

end