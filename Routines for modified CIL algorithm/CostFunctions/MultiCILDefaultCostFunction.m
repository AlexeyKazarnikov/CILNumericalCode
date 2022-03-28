classdef MultiCILDefaultCostFunction < CostFunction
    
    properties
        CIL = struct();
        Curve = struct();
        
        Ntr = 1000;
             
        Yave = {};
        C = [];
        iC = [];
        Np = {};
        ind1 = {};
        ind2 = {};
        
        R0 = {};
        base = {};
        nr = {};
		
        isUniformDist = false;
        experimental = false;
        
        DistanceProviders = {};
        
    end
    
    methods
        function obj = MultiCILDefaultCostFunction(cil,curve,providers)   
			obj.Abbreviation = 'MultiCILDefault';
            
            obj.CIL = cil;
            obj.Curve = curve;
            
            % CIL distribution data
            obj.Yave = cil.Yave;
            if isfield(cil,'iC')
                obj.iC = cil.iC;
            else
                obj.C = cil.C;
            end
            
            if isfield(cil,'experimental')
                obj.experimental = true;
            end
            
            obj.Np = cil.Np;
            obj.ind1 = cil.ind1;
            obj.ind2 = cil.ind2;
            
            % curve settings
            obj.R0 = curve.R0;
            obj.base = curve.base;
            obj.nr = curve.nr;           
            obj.isUniformDist = curve.isUniformDist;
            
            % distance providers
            obj.DistanceProviders = providers;
        end
        
        function [res,data] = evaluate(obj,varargin)
            [model,theta,S0] = check_evaluate_input(obj,varargin);
                 
            if ~obj.experimental
                S = model.simulate(obj.Np,[],theta);
            else
                S = model.simulate(2*obj.Np,[],theta);
            end
            
            if isa(S,'single')
                S0 = single(S0);
            end
            
            % early rejection test
            for k = 1:model.dim()
                Sk = model.select_dimensions(S,k);
                idxk = max(abs(Sk - repmat(mean(Sk,1),size(Sk,1),1))) < 1e-6;
                if sum(idxk) > 0
                    res = 1e15;
                    data = [];
                    return;
                end
            end
                     
            Nfeat = length(obj.DistanceProviders);
            Yf = [];
            for nf=1:Nfeat
                if ~obj.experimental
                    D = obj.DistanceProviders{nf}.distance(S0,S,model);
                else
                    S1 = S(:,1:obj.Np);
                    S2 = S(:,obj.Np+1:end);
                    D = obj.DistanceProviders{nf}.distance(S1,S2,model);
                end
                [Yref,~,~]=...
                  nlCreateECDF( ...
                  D, ...
                  obj.nr, ...
                  obj.isUniformDist, ...
                  obj.R0(nf), ...
                  obj.base(nf));
                Yref = Yref(obj.ind1(nf):obj.ind2(nf));
                Yf = [Yf Yref];
            end
            
            if ~isempty(obj.iC)
                res = (Yf-obj.Yave)*(obj.iC*(Yf-obj.Yave)');
            else
                res = (Yf-obj.Yave)*(obj.C \ (Yf-obj.Yave)');
            end
            
            data.Yref = Yf;                                
        end
        
        function f = visualize(obj,data)
            hold on
            plot(data.Yref,'bo-');
            plot(obj.Yave,'ro-');
            hold off
        end
        
        function data = serialize(obj)
            data.Abbreviation = obj.Abbreviation;
            data.Ntr = obj.Ntr;     
            data.CIL = obj.CIL;
            data.Curve = obj.Curve;
            data.DistanceProviders = {};
            for k=1:length(obj.DistanceProviders)
                data.DistanceProviders{k} = ...
                    obj.DistanceProviders{k}.serialize();
            end
        end
    end
    
    methods (Static)
        function obj = load(data)
            distance_providers = cell(1,length(data.DistanceProviders));
            for k=1:length(data.DistanceProviders)
                distance_providers{k} = ...
                    DistanceProvider.restore(data.DistanceProviders{k});
            end
            
            obj = MultiCILDefaultCostFunction( ...
                data.CIL, ...
                data.Curve, ...
                distance_providers ...
                );
            obj.Ntr = data.Ntr;
            obj.Abbreviation = data.Abbreviation;
        end
    end
end

