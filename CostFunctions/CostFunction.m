classdef CostFunction < handle
%   This abstract class defines a cost function, which could be used by the 
%   main scripts (CIL_simu.m, CIL_de.m and CIL_mcmc.m). Currently there is
%   only one implementation provided (CILCostFunction), but other cost 
%   functions could be implemented in similar manner as well.
    properties

    end
    
    methods (Abstract)
        % This method evaluates cost function for given model, vector of
        % control parameters and reference pattern dataset.
        %   INPUT
        %   model : reaction-diffunion model (expected to be an instance of 
        % subclass, derived from Model abstract class)
        %   theta : struct, containing the values of control parameters as
        % fields
        %   S0 : reference pattern dataset (stored in row=major format)
        %   OUTPUT
        %   res : the output of the cost function
        %   data : structure, which could be used to visualize the result
        [res,data] = evaluate(obj,model,theta,S0);        
        
        % This method is used to visualize the data, returned by the 
        % obj.evaluate() method.
        %   INPUT
        %   data : structure, returned by obj.evaluate()
        %   OUTPUT
        %   f : MATLAB graphical object
        f = show(obj,data);            
    end
end

