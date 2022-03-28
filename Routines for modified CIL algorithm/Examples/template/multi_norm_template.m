% required parameters

if ~exist('model','var')
    error("Please define model ('model' variable) before running the script!'");
end

if ~exist('N','var')
    error("Please define number of reference patterns ('N' variable) before running the script!");
end

if ~exist('theta0','var')
    error("Please define control parameter vector ('theta0' variable) before running the script!");
end

if ~exist('params','var')
    error("Please define sampling parameter names ('params' variable) before running the script!");
end

if ~exist('Nchain','var')
    error("Please define chain length ('Nchain' variable) before running the script!");
end

if ~exist('mcmc_flag','var')
    error("Please define MCMC chain flag ('mcmc_flag' variable) before running the script!");
end

% optional parameters

if ~exist('M','var')
    M = 12;
else
    fprintf('Using CIL dimension M=%i \n',M);
end

if ~exist('UseUniformSpacing','var')
    UseUniformSpacing = false;
else
    fprintf('Using uniform spacing %i \n',UseUniformSpacing);
end

if ~exist('ExperimentPrefix','var')
    ExperimentPrefix = '';
else
    fprintf('Using experiment filename prefix %s \n',ExperimentPrefix);
end

if ~exist('default_flag','var')
    default_flag = false;
else
    fprintf('Using default CIL builder %i \n',default_flag);
end

disp('Multi-CIL experiment')

ExperimentName = sprintf( ...
    '%s_multi_norm_pattern_%s_%s_%i', ...
    ExperimentPrefix, ...
    model.Abbreviation, ...
    model.Observer.Abbreviation, ...
    N ...
    );

fprintf('Experiment abbreviation: %s \n', ExperimentName);

storage = StorageHelper('Experiments', ExperimentName);

Nrun = 500;

distance_providers = {};

if strcmp(model.Observer.Abbreviation,'MinMaxObserver')
    distance_providers{1} = LpDistance(2);
    distance_providers{2} = W1pDistance(2);
    distance_providers{3} = W1pPrimeDistance(2);

    distance_prefix = {'L2','W12','W12prime'};
else
    distance_providers{1} = LpDistance(2);
    distance_providers{2} = LpDistance(0);
    distance_providers{3} = W1pDistance(2);
    distance_providers{4} = W1pDistance(0);
    distance_providers{5} = W1pPrimeDistance(2);
    distance_providers{6} = W1pPrimeDistance(0);

    distance_prefix = {'L2','Linf','W12','W1inf','W12prime','W1infprime'};
end

clear S

if ~mcmc_flag  
    % starting the experiment
    disp('Beginning the experiment...')

    % first we determine if there exist CIL distributions
    disp('Beginning CIL generation...')
    for k=1:length(distance_providers)
        CIL_file_name = strcat(distance_prefix{k},'_CIL.mat');
        CIL_file_path = storage.createLocalPath(CIL_file_name);
        if ~exist(CIL_file_path,'file')
            fprintf( ...
                'CIL for distance provider %s not found, generating data...\n', ...
                distance_prefix{k} ...
                    );
            if ~exist('S','var')
                Pattern_data_file_name = 'pattern_data.mat';
                Pattern_data_file_path = ...
                    storage.createLocalPath(Pattern_data_file_name);
                
                if ~exist(Pattern_data_file_path,'file')          
                    disp('Generating pattern data...')
                    if ~default_flag
                        S = model.simulate(N);
                    else
                        s1 = model.simulate(N);
                        s2 = model.simulate(N);
                        S = zeros(size(s1,1),size(s1,2),2);
                        S(:,:,1) = s1;
                        S(:,:,2) = s2;
                    end
                    disp('Saving pattern data to experiment folder...');
                    save(Pattern_data_file_path,'S');
                    disp('Done!')
                else
                    disp('Loading pattern data from experiment folder...');
                    load(Pattern_data_file_path);
                    disp('Done!')
                end
            end

            model.DistanceProvider = distance_providers{k};
            if ~default_flag
                builder = NewCILBuilder(model);
            else
                builder = DefaultCILBuilder(model);
            end
            builder.CIL.N = N;
            builder.CIL.M = M;
            builder.Settings.UseUniformSpacing = UseUniformSpacing;
            builder.estimate_ecdf_constants(theta0,S);
%            [c0,x0] = builder.generate_ecdf(theta0,S);

%             figure(1)
%             clf
%             plot(x0,c0,'ro-')
%             title(distance_prefix{k})
% 
%             pause

            builder.S = S;

            builder.generate_distribution(false);
            figure(2)
            clf
            plot(builder.Y','bo-')
            title(distance_prefix{k})

            pause

            figure(3)
            clf
            builder.perform_chi2_test();
            
            CIL_dim = builder.Chi2.rngindex2 - builder.Chi2.rngindex1;
            title(sprintf('%s (M=%i)',distance_prefix{k},CIL_dim))

            pause

            builder.estimate_distribution_parameters();
            data = builder.serialize();

            save(CIL_file_path, 'data');

            disp('CIL generation completed!')
        else
            fprintf( ...
                'CIL for distance provider %s found, skipping the generation... \n', ...
                distance_prefix{k} ...
                    );
        end
    end

    disp('All separate CIL distributions completed!')

    CIL_file_name = 'joint_CIL.mat';
    CIL_file_path = storage.createLocalPath(CIL_file_name);
    if ~exist(CIL_file_path,'file')
        disp('Joint CIL distribution not found, generating data...');

        if ~default_flag
            error('Not implemented yet!');
        else
            builder = MultiCILBuilder(model,distance_providers);
        end
            
        builder.CIL.N = N;
        builder.CIL.M = M;
        builder.Settings.UseUniformSpacing = UseUniformSpacing;
        
        if ~exist('S','var')
            Pattern_data_file_name = 'pattern_data.mat';
            Pattern_data_file_path = ...
                storage.createLocalPath(Pattern_data_file_name);

            if ~exist(Pattern_data_file_path,'file')          
                disp('Generating pattern data...')
                if ~default_flag
                    S = model.simulate(N);
                else
                    s = model.simulate(2*N);
                    s1 = s(:,1:N);
                    s2 = s(:,N+1:end);
                    S = zeros(size(s1,1),size(s1,2),2);
                    S(:,:,1) = s1;
                    S(:,:,2) = s2;
                end
                disp('Saving pattern data to experiment folder...');
                save(Pattern_data_file_path,'S');
                disp('Done!')
            else
                disp('Loading pattern data from experiment folder...');
                load(Pattern_data_file_path);
                disp('Done!')
            end
        end
        
        builder.estimate_ecdf_constants(theta0,S);
        [c0,x0] = builder.generate_ecdf(theta0,S);

        figure(1)
        clf
        for nf=1:length(distance_providers)
            subplot(1,length(distance_providers),nf)
            plot(x0{nf},c0{nf},'ro-')
            title('Joint')
        end

        pause

        builder.S = S;

        builder.generate_distribution(false);
        figure(2)
        clf
        for nf=1:length(distance_providers)
            subplot(1,length(distance_providers),nf)
            plot(builder.Y(:,:,nf)','bo-')
            title('Joint')
        end

        pause

        figure(3)
        clf
        builder.perform_chi2_test();
        
        CIL_dim = sum(builder.Chi2.rngindex2 - builder.Chi2.rngindex1);
        title(sprintf('Joint (M=%i)',CIL_dim))

        pause

        builder.estimate_distribution_parameters();
        data = builder.serialize();

        save(CIL_file_path, 'data');
    else
        disp('Joint CIL distribution found, skipping the CIL generation...');
    end

end

if ~mcmc_flag
    disp('Skipping MCMC generation...')
    return
end

disp('Beginning MCMC simulations...')

theta_start = [];
for i=1:size(params,1)
    theta_start(i) = params{i}{2};
end

for k=1:length(distance_providers)
    CIL_file_name = strcat(distance_prefix{k},'_CIL.mat');
    CIL_file_path = storage.createLocalPath(CIL_file_name);
    CIL_data = load(CIL_file_path);
    
    MCMC_file_name = strcat(distance_prefix{k},'_MCMC.mat');
    MCMC_file_path = storage.createLocalPath(MCMC_file_name);
    
    MCMC_fig_file_name = strcat(distance_prefix{k},'_MCMC_Fig%i.jpg');
    MCMC_fig_file_path = {};
    for f=1:4
        MCMC_fig_file_path{f} = ...
            storage.createLocalPath(sprintf(MCMC_fig_file_name,f));
    end
    
    if ~exist(MCMC_file_path,'file')
        runner = MCMCRunner(CIL_data.data, params);
    else
        MCMC_data = load(MCMC_file_path);
        runner = MCMCRunner.restore(MCMC_data.data);
    end
    
    if length(runner.Chain) < Nchain
        fprintf( ...
            'MCMC chain for distance provider %s is not completed, working...\n',...
            distance_prefix{k} ...
                );
     
        tic
        fprintf('%s: Value of the cost function for staring point: %.2f\n',...
                distance_prefix{k}, ...
                runner.ss_fun(theta_start,CIL_data.data) ...
                );
        t1=toc;
        
        fprintf('Cost function evaluation time: %f sec.\n',t1);
        
        while length(runner.Chain) < Nchain
            fprintf('%s: Running MCMC simulation: ready %i / %i \n', ...
                distance_prefix{k}, ...
                length(runner.Chain), ...
                Nchain ...
                );
            runner.run(Nrun,CIL_data.data);
            disp('Iteration completed!')
            data = runner.serialize();
            save(MCMC_file_path,'data');
            disp('Data saved!')
            
            figs = runner.plot_chain([1 2 3 4]);
            for f=1:4
                print(figs{f},MCMC_fig_file_path{f},'-djpeg');
            end
        end
        
        disp('MCMC chain completed!')
    else
        fprintf( ...
            'MCMC chain for distance provider %s completed, skipping...\n',...
            distance_prefix{k} ...
                ); 
    end    
end

CIL_file_name = 'joint_CIL.mat';
CIL_file_path = storage.createLocalPath(CIL_file_name);
CIL_data = load(CIL_file_path);

MCMC_file_name = 'joint_MCMC.mat';
MCMC_file_path = storage.createLocalPath(MCMC_file_name);

MCMC_fig_file_name = 'joint_Fig%i.jpg';
MCMC_fig_file_path = {};
for f=1:4
    MCMC_fig_file_path{f} = ...
        storage.createLocalPath(sprintf(MCMC_fig_file_name,f));
end

if ~exist(MCMC_file_path,'file')
    runner = MCMCRunner(CIL_data.data,params);
else
    MCMC_data = load(MCMC_file_path);
    runner = MCMCRunner.restore(MCMC_data.data);
end

if length(runner.Chain) < Nchain
    disp('Joint MCMC chain is not completed, working...')
    
    tic
    fprintf('Joint: Value of the cost function for staring point: %.2f\n', ...
            runner.ss_fun(theta_start,CIL_data.data) ...
            );
    t1=toc;

    fprintf('Cost function evaluation time: %f sec.\n',t1);

    while length(runner.Chain) < Nchain
        fprintf('Joint: Running MCMC simulation: ready %i / %i \n', ...
            length(runner.Chain), ...
            Nchain ...
            );
        runner.run(Nrun,CIL_data.data);
        disp('Iteration completed!')
        data = runner.serialize();
        save(MCMC_file_path,'data');
        disp('Data saved!')

        figs = runner.plot_chain([1 2 3 4]);
        for f=1:4
            print(figs{f},MCMC_fig_file_path{f},'-djpeg');
        end
    end

    disp('MCMC chain completed!')
else
    disp('Joint MCMC chain completed, skipping...')
end    

disp('All done!')
