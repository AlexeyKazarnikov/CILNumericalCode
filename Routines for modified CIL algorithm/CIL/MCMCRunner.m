% This class is used for the generation of MCMC chain, which corresponds to
% the CIL likelihood, created by 'DefaultCILBuilder' or 'MultiCILBuilder'
% classes.

classdef MCMCRunner < handle
    
    properties      
        Model = struct();
        MCMCModel = struct();
        MCMCOptions = struct();
        MCMCParams = {};
        MCMCRes = [];
        CostFunction = {};
        Chain = [];
        SumOfSquares = [];
    end
    
    methods
        function [res,data] = ss_fun(obj,theta,data)
            par_values = struct();
            for k=1:length(theta)
                par_values = ...
                    setfield(par_values,obj.MCMCParams{k}{1},theta(k));
            end
            
            k0 = randi(size(data.S,3));
            
            try
                [res,data] = obj.CostFunction.evaluate( ...
                    obj.Model, ...
                    par_values, ...
                    data.S(:,:,k0) ...
                    );
            catch ME
                warning('An exception occured while using function. Assigning res=1e15');
                disp(theta)
                res=1e15;
            end
        end
        
        function obj = MCMCRunner(data,params)
            if nargin == 0
                return;
            end
            
            obj.Model = Model.restore(data.Model);
            
            obj.MCMCModel.ssfun  = @(theta,data) obj.ss_fun(theta,data);
            obj.MCMCModel.sigma2 = 1;
            
            obj.MCMCOptions.qcov = 1e-8*eye(size(obj.MCMCParams,1));
            obj.MCMCOptions.verbosity=1;
            obj.MCMCOptions.adaptint = 300;
            if isunix
                obj.MCMCOptions.waitbar = 0;
            end
            
            obj.MCMCParams = params;
                   
            if strcmp(data.Type, 'Radon')
                obj.CostFunction = ...
                    TomographyDefaultCostFunction(data);
            elseif strcmp(data.Type, 'Default')
                cil = nlExtractCILData(data);
                curve = nlExtractCurveData(data);
                obj.CostFunction = ...
                    CILDefaultCostFunction( ...
                        cil, ...
                        curve ...
                        );
            elseif strcmp(data.Type, 'New')
                obj.CostFunction = ...
                    CILNewCostFunction( ...
                        data ...
                        );
            elseif strcmp(data.Type, 'MultiCIL')
                cil = nlExtractCILData(data);
                curve = nlExtractCurveData(data);
                distance_providers = cell(1,length(data.DistanceProviders));
                for k=1:length(data.DistanceProviders)
                    distance_providers{k} = ...
                        DistanceProvider.restore(data.DistanceProviders{k});
                end
                
                obj.CostFunction = ...
                    MultiCILDefaultCostFunction( ...
                        cil, ...
                        curve, ...
                        distance_providers ...
                        );
            elseif strcmp(data.Type, 'Synthetic')
                obj.CostFunction = ...
                    CILSyntheticCostFunction.restore(data.CostFunction);
            else
                error('Data is not recognised!');
            end                    
        end
        
        function [chain,s2,ss] = run(obj,num_elements,data)
            obj.MCMCOptions.nsimu = num_elements;
            obj.MCMCModel.N = num_elements;
            
            [obj.MCMCRes,chain,s2,ss] = ...
                mcmcrun( ...
                obj.MCMCModel, ...
                data, ...
                obj.MCMCParams, ...
                obj.MCMCOptions, ...
                obj.MCMCRes ...
                );
            obj.Chain = [obj.Chain; chain];
            obj.SumOfSquares = [obj.SumOfSquares; ss];
        end
        
        function data = serialize(obj)
            data.Model = ...
                obj.Model.serialize();
            data.MCMCModel = obj.MCMCModel;
            data.MCMCOptions = obj.MCMCOptions;
            data.MCMCParams = obj.MCMCParams;
            data.MCMCRes = obj.MCMCRes;
            data.CostFunction = ...
                obj.CostFunction.serialize();
            data.Chain = obj.Chain;
            data.SumOfSquares = obj.SumOfSquares;
        end
        
        function f = plot_chain(obj,fig_num,fig_type)
            if nargin < 3
                f{1}=figure(fig_num(1)); clf
                mcmcplot(obj.Chain,[],obj.MCMCRes.names,'chainpanel')

                f{2}=figure(fig_num(2)); clf
                mcmcplot(obj.Chain,[],obj.MCMCRes,'denspanel',2);

                f{3}=figure(fig_num(3)); clf
                mcmcplot(obj.Chain,[],obj.MCMCRes,'pairs');

                f{4}=figure(fig_num(4)); clf
                mcmcplot(obj.Chain,[],obj.MCMCRes,'pairs',2);
            else
                f=figure(fig_num(1)); clf
                mcmcplot(obj.Chain,[],obj.MCMCRes.names,fig_type);
            end
        end
    end
    
    methods (Static)
        function obj = restore(data)
            obj = MCMCRunner();
            obj.Model = Model.restore(data.Model);
            obj.MCMCModel = data.MCMCModel;
            obj.MCMCOptions = data.MCMCOptions;
            obj.MCMCParams = data.MCMCParams;
            obj.MCMCRes = data.MCMCRes;
            obj.CostFunction = ...
                CostFunction.restore(data.CostFunction);
            obj.Chain = data.Chain;
            obj.SumOfSquares = data.SumOfSquares;
        end
    end
end

