classdef DEOptimizer
    %DEOptimizer implements the Differential Evolution algorithm for 
    % stochastic optimization. It initialises with a given population 
    % and optional objective values, and provides methods to minimize an 
    % objective function while considering constraints and logging
    % progress, and to output some helpful information.
    
    properties
        Settings = struct();
    end
    
    properties (SetAccess = private)
        Population = [];
        ObjectiveValues = [];
    end
    
    methods
        function obj = DEOptimizer(population, objectives)
            if ~ismatrix(population) || ndims(population)~=2 || isempty(population)
                error('Population must be a non-empty 2D matrix!');
            end
            obj.Population = population;
            
            if nargin < 2 || isempty(objectives)
                obj.ObjectiveValues = [];
            else
                if ~ismatrix(objectives)
                    error('Objective values must be a matrix!');
                end
                obj.ObjectiveValues = objectives(:);
            end
            
            obj.Settings.MutationRate = 0.8;
            obj.Settings.CrossoverRate = 0.9;
            obj.Settings.ObjectiveRecalculation = false;
            obj.Settings.ShowProgress = true;
            obj.Settings.UseObjectiveVectorisation = false;           
        end
        
        function minimize(obj, obj_fun, constraint_fun, logFun)
            N = size(obj.Population,2);
            D = size(obj.Population,1);

            useLogging = nargin > 3 && ~isempty(logFun);
            
            if isempty(obj.ObjectiveValues)

                if useLogging
                    logFun('DE optimiser: BEGIN OBJECTIVE INIT');
                end

                obj.ObjectiveValues = zeros(1,N);
                if ~obj.Settings.UseObjectiveVectorisation
                    for k=1:N
                        if useLogging
                            logFun( ...
                                sprintf( ...
                                'DE optimiser: OBJECTIVE INIT (%i / %i)', k, N) ...
                                );
                        end
                        obj.ObjectiveValues(k) = obj_fun(obj.Population(:,k));
                        if obj.Settings.ShowProgress
                            fprintf('Objectives initialisation: %i / %i \n', k, N);
                        end
                    end
                else
                    if useLogging
                            logFun( ...
                                'DE optimiser: OBJECTIVE INIT (vectorised)' ...
                                );
                    end
                    obj.ObjectiveValues = obj_fun(obj.Population);
                    if obj.Settings.ShowProgress
                        disp('Objectives initialisation (vectorised)');
                    end
                end

                if useLogging
                    logFun('DE optimiser: END OBJECTIVE INIT');
                end
            end
            
            [y_best,~,ind_best] = obj.best_candidate();

            if useLogging
                logFun( ...
                    sprintf( ...
                    'DE optimiser: BEST CANDIDATE: %i', ind_best) ...
                    );
            end
        
            % we construct a permutation matrix to create unique pairs
            permat = zeros(N,2);
            for k=1:N
                allowed_indices = 1:N;
                allowed_indices([k ind_best]) = [];
                permat(k,:) = allowed_indices( ...
                    randperm(length(allowed_indices),2));
            end

            % generate donors by mutation
            donor_elements = repmat(y_best,1,N) + ...
                obj.Settings.MutationRate * ...
                (obj.Population(:,permat(:,1))-obj.Population(:,permat(:,2)) ...
                );

            % perform recombination
            r=repmat(randi(D,1,N),D,1);
            basej=repmat(1:D,N,1)'; %used later
            muv = ((rand(D,N)<obj.Settings.CrossoverRate) | (basej==r));
            mux = 1-muv;
            new_candidates = obj.Population.*mux + donor_elements.*muv;

            % constraint check
            if nargin > 2 && ~isempty(constraint_fun)
                new_candidates = constraint_fun(new_candidates);
            end

            % greedy selection
            if useLogging
                logFun('DE optimiser: BEGIN OBJECTIVE UPDATE');
            end

            new_objective_values = zeros(1,N);
            if ~obj.Settings.UseObjectiveVectorisation
                for k=1:N
                    if useLogging
                        logFun( ...
                            sprintf( ...
                            'DE optimiser: OBJECTIVE UPDATE (%i / %i)', k, N) ...
                            );
                    end
                    
                    new_objective_values(k) = obj_fun(new_candidates(:,k));
                    if obj.Settings.ShowProgress
                        fprintf('Objectives update: %i / %i \n', k, N);
                    end
                end
            else
                if useLogging
                    logFun( ...
                        'DE optimiser: OBJECTIVE UPDATE (vectorised)' ...
                        );
                end
                new_objective_values = obj_fun(new_candidates);
                if obj.Settings.ShowProgress
                    disp('Objectives update (vectorised)');
                end
            end

            if useLogging
                logFun('DE optimiser: END OBJECTIVE UPDATE');
            end

            idx = new_objective_values < obj.ObjectiveValues;
            obj.Population(:,idx)=new_candidates(:,idx);
            
            if ~obj.Settings.ObjectiveRecalculation
                obj.ObjectiveValues(idx)=new_objective_values(idx);
            else
                if ~obj.Settings.UseObjectiveVectorisation
                    for k=1:N
                        if useLogging
                            logFun('DE optimiser: BEGIN OBJECTIVE RECALCULATION');
                        end
    
                        if useLogging
                            logFun( ...
                                sprintf( ...
                                'DE optimiser: OBJECTIVE RECALCULATION (%i / %i)', k, N) ...
                                );
                        end
    
                        obj.ObjectiveValues(k) = obj_fun(obj.Population(:,k));
                        if obj.Settings.ShowProgress
                            fprintf('Objectives recalculation: %i / %i \n', k, N);
                        end
                    end
                else
                    if useLogging
                        logFun( ...
                            'DE optimiser: OBJECTIVE RECALCULATION (vectorised)' ...
                            );
                    end
                    obj.ObjectiveValues = obj_fun(obj.Population);
                    if obj.Settings.ShowProgress
                        disp('Objectives recalculation (vectorised)');
                    end
                end

                if useLogging
                    logFun('DE optimiser: END OBJECTIVE RECALCULATION');
                end
            end
        end
        
        function [val,fval,ind] = best_candidate(obj)
            [fval,ind] = min(obj.ObjectiveValues);
            val = obj.Population(:,ind);
        end
        
        function S = population(obj)
            S = obj.Population;
        end
        
        function fval = objectives(obj)
            fval = obj.ObjectiveValues;
        end
        
        function mu = population_mean(obj)
            mu = mean(obj.Population,2);
        end
        
        function sigma = population_std(obj)
            sigma = std(obj.Population,[],2);
        end
    end
    
end

