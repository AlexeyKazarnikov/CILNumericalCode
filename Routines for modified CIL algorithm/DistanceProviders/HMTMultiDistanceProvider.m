classdef HMTMultiDistanceProvider < DistanceProvider
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    methods
        function obj = HMTMultiDistanceProvider()
            obj.Abbreviation = 'HMT';
            obj.Settings.UseMinMaxSettings = false;
        end
        
        function data = serialize(obj)
            data.Abbreviation = obj.Abbreviation;
            data.Settings = obj.Settings;
        end
        
        function d = distance(obj,S1,S2,model)
            if isempty(S2)
                S2 = S1;
            end
            
            N = model.Settings.sol_par.grid_resolution;
                        
            data = model.serialize();
            data.Abbreviation = 'HMTBasic';
            model_basic = Model.restore(data);
            
            S1_basic = S1(N+1:end,:);                   
            S2_basic = S2(N+1:end,:);
            
            distance_providers = {};
            if ~obj.Settings.UseMinMaxSettings
                distance_providers{1} = LpDistance(2);
                distance_providers{2} = LpDistance(0);
                distance_providers{3} = W1pDistance(2);
                distance_providers{4} = W1pDistance(0);
                distance_providers{5} = W1pPrimeDistance(2);
                distance_providers{6} = W1pPrimeDistance(0);
            else
                distance_providers{1} = LpDistance(2);
                distance_providers{2} = W1pDistance(2);
                distance_providers{3} = W1pPrimeDistance(2);
            end
            
            D = length(distance_providers);
            
            D1 = 1;
            if ~obj.Settings.UseMinMaxSettings
                D1 = 2;
            end
            
            d = zeros(size(S1,2),size(S2,2),D);
                  
            for k=1:D1
                d(:,:,k) = distance_providers{k}.distance(S1,S2,model);
            end
            for k=D1+1:D
                d(:,:,k) = distance_providers{k}.distance( ...
                    S1_basic, ...
                    S2_basic, ...
                    model_basic ...
                    );
            end
        end
    end
    
    methods (Static)
        function obj = load(data)
            obj = HMTMultiDistanceProvider();    
            obj.Abbreviation = data.Abbreviation;
            obj.Settings = data.Settings;
        end
    end
end

