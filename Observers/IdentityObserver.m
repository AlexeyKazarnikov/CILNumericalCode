classdef IdentityObserver < Observer
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties

    end
    
    methods
        function obj = IdentityObserver()
         
        end
        
        function Sout = transform(obj,S,model)
            Sout = S;
        end
    end
end

