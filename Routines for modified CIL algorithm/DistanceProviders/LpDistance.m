classdef LpDistance < DistanceProvider
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    methods
        function obj = LpDistance(P)
            obj.Settings.P = P;        
            obj.Abbreviation = 'Lp';
        end
        
        function d = distance(obj,S1,S2,model)
            use_one_arg = isempty(S2);
            
            if isa(S1,'struct')
                vdim = S1.vdim;
                sdim = S1.sdim;
                spartprod = prod(S1.spart);
            else
                vdim = model.dim();
                sdim = model.size();
                spartprod = prod(model.dx());
            end
            
            N1 = size(S1,2);
            if ~use_one_arg
                N2 = size(S2,2);
            else
                N2 = N1;
            end
            
            d = zeros(N1,N2);
            
            if obj.Settings.P > 0
                [idx,coeff] = nlCreateBoundaryMask( ...
                    sdim, ...
                    obj.Settings.P ...
                    );
                coeff1 = repmat(coeff',1,N1);
                coeff2 = repmat(coeff',1,N2);
            end
                       
            for k = 1:vdim
                if isa(S1,'struct')
                    S1k = nlSelectDimensions(S1,k);
                    S1k = S1k.data;
                else
                    S1k = model.select_dimensions(S1,k);
                end
                if ~use_one_arg
                    if isa(S1,'struct')
                        S2k = nlSelectDimensions(S2,k);
                        S2k = S2k.data;
                    else
                        S2k = model.select_dimensions(S2,k);
                    end
                end

                if obj.Settings.P > 0
                    S1k(idx,:) = coeff1.*S1k(idx,:);
                    if ~use_one_arg
                        S2k(idx,:) = coeff2.*S2k(idx,:);
                    end
                end
                
                % TODO : fix that!
                if ~use_one_arg
                    dk = DistanceMatrixLpMEX(S1k,S2k,obj.Settings.P);
                else
                    dk = DistanceMatrixLpMEX(S1k,S1k,obj.Settings.P);
                end
                
                if (obj.Settings.P > 0)
                    dk = dk * spartprod^(1 / obj.Settings.P);
                end
                d = d + dk;
            end       
        end
        
        function data = serialize(obj)
            data.Abbreviation = obj.Abbreviation;
            data.Settings = obj.Settings;
        end
    end
    
    methods (Static)
        function obj = load(data)
            obj = LpDistance(data.Settings.P);    
            obj.Abbreviation = data.Abbreviation;
            obj.Settings = data.Settings;
        end
    end
end

