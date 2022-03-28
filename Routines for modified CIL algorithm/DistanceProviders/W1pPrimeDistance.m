classdef W1pPrimeDistance < DistanceProvider
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    methods
        function obj = W1pPrimeDistance(P)
            obj.Settings.P = P;
            obj.Abbreviation = 'W1pPrime';
        end
        
        function d = distance(obj,S1,S2,model)
            d = zeros(size(S1,2),size(S2,2));
            
            %dS1 = nlDiffPatterns(S1,model);
            %dS2 = nlDiffPatterns(S2,model);
            
            if strcmp(model.Abbreviation,'HMT')
                data = model.serialize();
                data.Abbreviation = 'HMTBasic';
                model = Model.restore(data);
                N = model.Settings.sol_par.grid_resolution;
                S1 = S1(N+1:end,:);
                S2 = S2(N+1:end,:);
            end
            
            dS1 = model.differentiate(S1);
            dS2 = model.differentiate(S2);
            
            if obj.Settings.P > 0
                [idx,coeff] = nlCreateBoundaryMask( ...
                    model.size(), ...
                    obj.Settings.P ...
                    );

                coeff1 = repmat(coeff',1,size(S1,2));
                coeff2 = repmat(coeff',1,size(S2,2));
            end
            
            for k = 1:model.dim()
                S1k = model.select_dimensions(S1,k);
                S2k = model.select_dimensions(S2,k);
                
                if obj.Settings.P > 0
                    S1k(idx,:) = coeff1.*S1k(idx,:);
                    S2k(idx,:) = coeff2.*S2k(idx,:);
                end
                
                dk = DistanceMatrixPowLpMEX(S1k,S2k,obj.Settings.P);
        
                if obj.Settings.P > 0
                    dk = dk * prod(model.dx());
                end
                
                for l=1:length(dS1)
                    dS1k = model.select_dimensions(dS1{l},k);
                    dS2k = model.select_dimensions(dS2{l},k);
                    
                    ddist = DistanceMatrixPowLpMEX(dS1k,dS2k,obj.Settings.P);
                    
                    if obj.Settings.P > 0
                        dk = dk + ddist * prod(model.dx());
                    else
                        dk = max(dk,ddist);
                    end
                end
                
                if obj.Settings.P > 1
                    dk = dk.^(1/obj.Settings.P);
                end
                
                d = d + dk;
            end             
        end
        
        function data = serialize(obj)
            data.Abbreviation = obj.Abbreviation;
            data.Settings = obj.Settings;
        end
    end
        
    methods (Static)
        function obj = load(data)
            obj = W1pPrimeDistance(data.Settings.P);    
            obj.Abbreviation = data.Abbreviation;
            obj.Settings = data.Settings;
        end
    end
end

