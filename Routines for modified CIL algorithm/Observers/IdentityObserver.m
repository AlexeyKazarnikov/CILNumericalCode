classdef IdentityObserver < Observer
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties

    end
    
    methods
        function obj = IdentityObserver()
            obj.Abbreviation = 'IdentityObserver';
        end
        
        function Sout = transform(obj,S,model)
            Sout = S;
        end
        
        function data = serialize(obj)
            data.Abbreviation = obj.Abbreviation;
        end
    end
    
    methods (Static)
        function obj = load(data)
            obj = IdentityObserver();    
            obj.Abbreviation = data.Abbreviation;
        end
    end
end

