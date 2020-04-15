classdef MinMaxObserver < Observer
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties

    end
    
    methods
        function obj = MinMaxObserver()
         
        end
        
        function Sout = transform(obj,S,model)
            Sout = model.normalize(S);
        end
    end
end
