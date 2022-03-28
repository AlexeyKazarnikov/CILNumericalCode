% This abstract class contains the definitions for properties and methods
% used by all CIL-based cost functions inside this library. In addition to
% abstract methods (which must be implemented in the derived classes), this
% base class contains helper method for argument validation.

classdef CostFunction < handle
    
    properties
        % all intermediate settings of the cost function must be stored in
        % this field
        Settings = struct(); 
        % each cost function must possess the unique abbreviation, which is
        % used during serialization. This value must be adjusted by the
        % derived classes
        Abbreviation = 'CostFunction';
    end
    
    methods (Abstract)
        % This method evaluates the cost function for the specified model,
        % parameter values and reference patterns.
        %   INPUT
        %   model : an instance of Model class
        %   theta : a struct, containing the values of varying model
        %   parameters in the respective fields
        %   S0 : 2D array, which contains reference pattern(s)
        %   OUTPUT
        %   res : an output of the cost function (scalar value)
        %   data : data, which could be used to visualize the results  
        [res,data] = evaluate(obj,model,theta,S0);
        
        % This method visualize the results of the obj.evaluate() call.
        %   INPUT
        %   data : data structure, produced by obj.evaluate()
        %   OUTPUT
        %   g : graphics handle
        g = visualize(obj,data);
        
        % This method serializes the object and saves the result into the
        % output structure
        %   OUTPUT
        %   data : structure, which contains the serialized object
        data = serialize(obj);
    end
    
    methods (Static)
        % This method de-serializes the object of the same class
        %   INPUT
        %   data : structure, which contains the serialized object
        %   OUTPUT
        %   obj : an instance of the respective class 
        function obj = load(data)
            error('It is impossible to load an abstract class! Use CostFunction.restore() instead.');
        end
        
        % This method de-serializes the object by recognising the
        % abbreviation and calling the respective method of the derived
        % class
        %   INPUT
        %   data : structure, which contains the serialized object
        %   OUTPUT
        %   obj : an instance of the respective class
        function obj = restore(data)
            if strcmp(data.Abbreviation,'CILDefault')
                obj = CILDefaultCostFunction.load(data);
            elseif strcmp(data.Abbreviation,'MultiCILDefault')
                obj = MultiCILDefaultCostFunction.load(data);
            elseif strcmp(data.Abbreviation,'TomographyDefault')
                obj = TomographyDefaultCostFunction.load(data);
            elseif strcmp(data.Abbreviation,'CILSynthetic')
                obj = CILSyntheticCostFunction.load(data);
            else
                error('CostFunction class was not recognised!');
            end
        end
    end
    
    methods
        % This helper method validates the input of the 'evaluate' method
        %   INPUT
        %   vararg : cell array, which contains the input arguments
        %   OUTPUT
        %   model : an instance of Model class, which has produced the
        %   pattern data being used ('S0' output parameter)
        %   theta : a struct, containing the values of varying model
        %   parameters in the respective fields
        %   S0 : 2D array, which contains reference pattern(s)   
        function [model,theta,S0] = check_evaluate_input(obj,vararg)
            model = nlExtractArg(vararg,1);
            theta = nlExtractArg(vararg,2);
            S0 = nlExtractArg(vararg,3);
            
            % checking for empty values
            if isempty(model)
                error('Input argument model must be set!');
            end
            if isempty(theta)
                theta = struct();
            end
            if isempty(S0)
                error('Input argument S0 must be set!');
            end
            
            % checking data types
            if ~isa(model, 'Model')
                error('Input argument model must be an instance of Model class!');
            end
            if ~isstruct(theta)
                error('Input argument theta must be a struct!');
            end
            if ~ismatrix(S0) || ndims(S0) ~= 2
                error('Input argument S0 must be a 2D matrix!');
            end
        end
    end
end

