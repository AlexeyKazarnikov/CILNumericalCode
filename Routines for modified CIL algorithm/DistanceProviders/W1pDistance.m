classdef W1pDistance < DistanceProvider
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    methods
        function obj = W1pDistance(P)
            obj.Settings.P = P;
            obj.Abbreviation = 'W1p';
        end
        
        function d = distance(obj,S1,S2,model)
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
            
            if isa(S1,'struct')
                dS1 = nlDiffPatterns(S1);
                dS2 = nlDiffPatterns(S2);
            else
                dS1 = model.differentiate(S1);
                dS2 = model.differentiate(S2);
            end
            
            Lp_norm = LpDistance(obj.Settings.P);
            
            if isa(S1,'struct')
                d = Lp_norm.distance(S1,S2);
            else
                d = Lp_norm.distance(S1,S2,model);
            end
            
            for k=1:length(dS1)
                if isa(S1,'struct')
                    d = d + Lp_norm.distance(dS1{k},dS2{k});
                else
                    d = d + Lp_norm.distance(dS1{k},dS2{k},model);
                end
            end  
        end
        
        function data = serialize(obj)
            data.Abbreviation = obj.Abbreviation;
            data.Settings = obj.Settings;
        end
    end
    
    methods (Static)
        function obj = load(data)
            obj = W1pDistance(data.Settings.P);    
            obj.Abbreviation = data.Abbreviation;
            obj.Settings = data.Settings;
        end
    end
end

