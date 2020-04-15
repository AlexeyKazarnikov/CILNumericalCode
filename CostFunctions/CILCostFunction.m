classdef CILCostFunction < CostFunction
% This class provides the implementation for the CIL cost function, defined
% in the paper.
    
    properties
        % CIL distribution settings
        Yave = []; % mean vector
        iC = []; % inverse covariance matrix
        Np = 0; % number of patterns in the test pattern set
        ind1 = 0; % start 'tail' index
        ind2 = 0; % end 'tail' index
        
        % eCDF vector settings
        R0 = 0; % maximal radius R0
        base = 0; % decay rate
        nr = 0; % number of bins (the dimension of eCDF vector)        
        
    end
    
    methods
        function obj = CILCostFunction(cil,curve)        
            % CIL distribution data
            obj.Yave = cil.Yave;
            obj.iC = cil.iC;
            obj.Np = cil.N;
            obj.ind1 = cil.ind1;
            obj.ind2 = cil.ind2;
            
            % curve settings
            obj.R0 = curve.R0;
            obj.base = curve.base;
            obj.nr = curve.nr;           
        end
        
        function [res,data] = evaluate(obj,model,theta,S0)
            % generating test pattern set
            S = model.simulate(obj.Np,[],theta);
            
            % computing distance between test pattern set and reference
            % pattern set
            D = model.distance(S0,S);
            
            % computing a test eCDF vector
            [Yref,~,~]=...
                  nlCreateECDF(D,obj.nr,false,obj.R0,obj.base);
            
            % cutting 'tails'
            Yref = Yref(obj.ind1:obj.ind2);
            
            % evaluating the result
            res = (Yref-obj.Yave)*(obj.iC*(Yref-obj.Yave)');
            
            % filling the data structure
            data.Yref = Yref;                                
        end
        
        function f = show(obj,data)
            hold on
            f{1} = plot(data.Yref,'bo-');
            f{2} = plot(obj.Yave,'ro-');
            hold off
        end
    end
end

