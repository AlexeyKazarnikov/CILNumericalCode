classdef MinMaxObserver < Observer
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties

    end
    
    methods
        function obj = MinMaxObserver()
            obj.Abbreviation = 'MinMaxObserver';
        end
        
        function Sout = transform(obj,S,model)
            Sout = model.normalize(S);
        end
        
        function data = serialize(obj)
            data.Abbreviation = obj.Abbreviation;
        end
    end
    
    methods (Static)
        function obj = load(data)
            obj = MinMaxObserver();    
            obj.Abbreviation = data.Abbreviation;
        end
    end
end
