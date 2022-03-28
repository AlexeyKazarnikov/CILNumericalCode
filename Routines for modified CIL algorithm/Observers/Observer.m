% This abstract class contains the definitions for properties and methods
% used by all observers inside this library. All abstract methods, defined
% here, must be implemented in the derived classes.

classdef Observer < handle
    
    properties
        % each observer must possess the unique abbreviation, which is
        % used during serialization. This value must be adjusted by the
        % derived classes
        Abbreviation = 'Observer';
    end
    
    methods (Static)
        % This method de-serializes the object of the same class
        %   INPUT
        %   data : structure, which contains the serialized object
        %   OUTPUT
        %   obj : an instance of the respective class 
        function obj = load(data)
            error('It is impossible to load an abstract class! Use Observer.restore() instead.');
        end
        
        % This method de-serializes the object by recognising the
        % abbreviation and calling the respective method of the derived
        % class
        %   INPUT
        %   data : structure, which contains the serialized object
        %   OUTPUT
        %   obj : an instance of the respective class
        function obj = restore(data)
            if strcmp(data.Abbreviation,'IdentityObserver')
                obj = IdentityObserver.load(data);
            elseif strcmp(data.Abbreviation,'MinMaxObserver')
                obj = MinMaxObserver.load(data);
            else
                error('Observer class was not recognised!');
            end
        end
    end
    
    methods
        % This method implements the observation transformation of the
        % pattern data
        %   INPUT
        %   S : pattern data (2D array)
        %   model : a model, which has produced this pattern data
        %   OUTPUT
        %   Sout : transformed pattern data (2D array)
        function Sout = transform(obj,S,model)
            error('The method is not implemented!');
        end
        
        % This method serializes the object and saves the result into the
        % output structure
        %   OUTPUT
        %   data : structure, which contains the serialized object
        function data = serialize(obj)
            error('The method is not implemented!');
        end
    end
end

