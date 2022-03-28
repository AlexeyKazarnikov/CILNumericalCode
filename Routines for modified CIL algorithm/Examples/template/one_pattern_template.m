if ~exist('model','var')
    error("Please define model ('model' variable) before running the script!'");
end

if ~exist('sref','var')
    error("Please define reference pattern data ('sref' variable) before running the script!");
end

if ~exist('cf','var')
    error("Please define cost function ('cf' variable) before running the script!");
end

if ~exist('params','var')
    error("Please define sampling parameter names ('params' variable) before running the script!");
end

if ~exist('Nchain','var')
    error("Please define chain length ('Nchain' variable) before running the script!");
end

disp('One pattern experiment')

exp_data.S = sref;
exp_data.Model = model.serialize();
exp_data.CostFunction = cf.serialize();
exp_data.Type = 'Synthetic';

ExperimentName = sprintf( ...
    'one_pattern_%s_%s_%i', ...
    model.Abbreviation, ...
    model.Observer.Abbreviation, ...
    size(sref,2) ...
    );

fprintf('Experiment abbreviation: %s \n', ExperimentName);

storage = StorageHelper('Experiments', ExperimentName);

data_file_path = storage.createLocalPath('exp_data.mat');
if ~exist(data_file_path,'file')
    disp('Experiment data has been copied to the output directory.')
    save(data_file_path,'exp_data');
else
    disp('Experiment data has been loaded from the output directory.')
    load(data_file_path);
end

MCMC_file_path = storage.createLocalPath('chain_MCMC.mat');

MCMC_fig_file_name = 'chain_Fig%i.jpg';
MCMC_fig_file_path = {};
for f=1:4
    MCMC_fig_file_path{f} = ...
        storage.createLocalPath(sprintf(MCMC_fig_file_name,f));
end

if ~exist(MCMC_file_path,'file')
    runner = MCMCRunner(exp_data,params);
else
    MCMC_data = load(MCMC_file_path);
    runner = MCMCRunner.restore(MCMC_data.data);
end

if length(runner.Chain) < Nchain
    disp('MCMC chain is not completed, working...')
    
    theta_start = [];
    for i=1:size(params,1)
        theta_start(i) = params{i}{2};
    end

    tic;
    fprintf('Value of the cost function for staring point: %.2f\n',...
            runner.ss_fun(theta_start,exp_data) ...
            );
    toc;

    while length(runner.Chain) < Nchain
        fprintf('Running MCMC simulation: ready %i / %i \n', ...
            length(runner.Chain), ...
            Nchain ...
            );
        runner.run(100,exp_data);
        disp('Iteration completed!')
        data = runner.serialize();
        save(MCMC_file_path,'data');
        disp('Data saved!')

        try
            figs = runner.plot_chain([1 2 3 4]);
            for f=1:4
                print(figs{f},MCMC_fig_file_path{f},'-djpeg');
            end
        catch ME
            warning('An error has occured while plotting the data!');
        end
    end

    disp('MCMC chain completed!')
else
    disp('MCMC chain completed, skipping...')
end

