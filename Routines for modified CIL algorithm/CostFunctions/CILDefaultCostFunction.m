classdef CILDefaultCostFunction < CostFunction
    
    properties
        CIL = struct();
        Curve = struct();
        
        Ntr = 1000;
              
        Yave = [];
        C = [];
        iC = [];
        Np = 0;
        ind1 = 0;
        ind2 = 0;
        
        R0 = 0;
        base = 0;
        nr = 0;
        
        isUniformDist = false;  
        experimental = false;
    end
    
    methods
        function obj = CILDefaultCostFunction(cil,curve)
            obj.Abbreviation = 'CILDefault';
            
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
        end
        
        function [res,data] = evaluate(obj,varargin)
            [model,theta,S0] = check_evaluate_input(obj,varargin);
            
            if ~obj.experimental
                S = model.simulate(obj.Np,[],theta);
            else
                S = model.simulate(2*obj.Np,[],theta);
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
            
            if ~obj.experimental
                D = model.distance(S0,S);
            else
                S1 = S(:,1:obj.Np);
                S2 = S(:,obj.Np+1:end);
                D = model.distance(S1,S2);
            end
            
            [Yref,~,~]=...
                  nlCreateECDF(D,obj.nr,obj.isUniformDist,obj.R0,obj.base);
            Yref = Yref(obj.ind1:obj.ind2);
            
            if isempty(obj.C)
                res = (Yref-obj.Yave)*(obj.iC*(Yref-obj.Yave)');
            else
                res = (Yref-obj.Yave)*(obj.C \ (Yref-obj.Yave)');
            end
            
            data.Yref = Yref;                                
        end
        
        function f = visualize(obj,data)
            hold on
            f{1} = plot(data.Yref,'bo-');
            f{2} = plot(obj.Yave,'ro-');
            hold off
        end
        
        function data = serialize(obj)
            data.Abbreviation = obj.Abbreviation;
            data.Ntr = obj.Ntr;     
            data.CIL = obj.CIL;
            data.Curve = obj.Curve;
        end
    end
    
    methods (Static)
        function obj = load(data)
            obj = CILDefaultCostFunction(data.CIL,data.Curve);
            obj.Ntr = data.Ntr;
            obj.Abbreviation = data.Abbreviation;
        end
    end
end

