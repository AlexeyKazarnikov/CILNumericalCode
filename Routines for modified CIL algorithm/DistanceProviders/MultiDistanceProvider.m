classdef MultiDistanceProvider < DistanceProvider
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    methods
        function obj = MultiDistanceProvider()
            obj.Abbreviation = 'MDP';
            obj.Settings.UseL2 = true;
            obj.Settings.UseLinf = true;
            obj.Settings.UseW12 = true;
            obj.Settings.UseW1inf = true;
            obj.Settings.UseW12Prime = true;
            obj.Settings.UseW1infPrime = true;
        end
        
        function res = get_distance_number(obj)
            res = 0;
            if obj.Settings.UseL2
                res = res + 1;
            end
            if obj.Settings.UseLinf
                res = res + 1;
            end
            if obj.Settings.UseW12
                res = res + 1;
            end
            if obj.Settings.UseW1inf
                res = res + 1;
            end
            if obj.Settings.UseW12Prime
                res = res + 1;
            end
            if obj.Settings.UseW1infPrime
                res = res + 1;
            end
        end
        
        function [dist_L2_sqr, dist_Linf_sqr] = compute_basic_distances( ...
            obj,S1,S2,model)  
            use_one_arg = isempty(S2);
            S1s = single(S1);
            if ~use_one_arg
                S2s = single(S2);
            else
                S2s = [];
            end
            
            N1 = size(S1,2);
            if ~use_one_arg
                N2 = size(S2,2);
            else
                N2 = N1;
            end
            
            D = model.dim();
        
            if (obj.Settings.UseL2)
                dist_L2_sqr = zeros(N1,N2,D);
                [idx2,coeff2] = ...
                    nlCreateBoundaryMask( ...
                        model.size(), ...
                        2 ...
                    );
            else
                dist_L2_sqr = [];
            end
            
            if (obj.Settings.UseLinf)
                dist_Linf_sqr = zeros(N1,N2,D);
            else
                dist_Linf_sqr = [];
            end
        
            for k = 1:D 
                S1k = model.select_dimensions(S1s,k);
                S1k0 = S1k;
                S1k2 = S1k;
                S1k2(idx2,:) = repmat(coeff2',1,N1).*S1k2(idx2,:);
                
                if ~use_one_arg
                    S2k = model.select_dimensions(S2s,k);
                    S2k0 = S2k;
                    S2k2 = S2k;
                    S2k2(idx2,:) = repmat(coeff2',1,N2).*S2k2(idx2,:);
                end
                
                if ~use_one_arg
                    if obj.Settings.UseL2
                        
%                         dist_L2_sqr(:,:,k) = ...
%                             prod(model.dx()) * ...
%                             DistanceMatrixPowLpMEX( ...
%                                 S1k2, ...
%                                 S2k2, ...
%                                 2 ...
%                             );
                        
                        dist_L2_sqr(:,:,k) = ...
                            prod(model.dx()) * ...
                            single(DistanceMatrixPowL2CUDA( ...
                                double(S1k2), ...
                                double(S2k2) ...
                            ));
                    end
                    
                    if obj.Settings.UseLinf
                        dist_Linf_sqr(:,:,k) = ...
                            DistanceMatrixPowLpMEX( ...
                                S1k0, ...
                                S2k0, ...
                                0 ...
                            );
                    end
                else
                    if obj.Settings.UseL2
                        
%                         dist_L2_sqr(:,:,k) = ...
%                             prod(model.dx()) * ...
%                             DistanceMatrixPowLpMEX( ...
%                                 S1k2, ...
%                                 2 ...
%                             );
                        
                        dist_L2_sqr(:,:,k) = ...
                            prod(model.dx()) * ...
                            single(DistanceMatrixPowL2CUDA( ...
                                double(S1k2), ...
                                double(S1k2) ...
                            ));
                    end
                    
                    if obj.Settings.UseLinf
                        dist_Linf_sqr(:,:,k) = ...
                            DistanceMatrixPowLpMEX( ...
                                S1k0, ...
                                0 ...
                            );
                    end
                end
            end % for
        end
        
        function d = distance(obj,S1,S2,model)
            use_one_arg = isempty(S2);
            
            dS1 = model.differentiate(S1);
            if ~use_one_arg
                dS2 = model.differentiate(S2);
            end
            K = length(dS1);
            
            S1s = single(S1);
            if ~use_one_arg
                S2s = single(S2);
            else
                S2s = [];
            end
            
            N1 = size(S1,2);
            if ~use_one_arg
                N2 = size(S2,2);
            else
                N2 = N1;
            end
            
            D = model.dim();        
            
            % first we compute basic norms (L2 and Linf)
            [dist_L2_sqr,dist_Linf_sqr] = ...
                obj.compute_basic_distances(S1s,S2s,model);
            
            if obj.Settings.UseL2
                dist_L2_sqr_deriv = zeros(N1,N2,D,K);
            else
                dist_L2_sqr_deriv = [];
            end
            if obj.Settings.UseLinf
                dist_Linf_sqr_deriv = zeros(N1,N2,D,K);
            else
                dist_Linf_sqr_deriv = [];
            end
            
            for k=1:K
                if ~use_one_arg
                    [dist_L2_sqr_deriv(:,:,:,k), dist_Linf_sqr_deriv(:,:,:,k)] = ...
                        obj.compute_basic_distances( ...
                            dS1{k}, ...
                            dS2{k}, ...
                            model ...
                        );
                else
                    [dist_L2_sqr_deriv(:,:,:,k), dist_Linf_sqr_deriv(:,:,:,k)] = ...
                        obj.compute_basic_distances( ...
                            dS1{k}, ...
                            [], ...
                            model ...
                        );
                end
            end
            
            % finally we assemble the required distances
            M = obj.get_distance_number();
            d = zeros(N1,N2,M);    
            counter = 1;
            
            if obj.Settings.UseL2
                for k=1:D
                    d(:,:,counter) = ...
                        d(:,:,counter) + sqrt(dist_L2_sqr(:,:,k));
                end
                counter = counter + 1;
            end
            
            if obj.Settings.UseLinf
                for k=1:D
                    d(:,:,counter) = ...
                        d(:,:,counter) + dist_Linf_sqr(:,:,k);
                end
                counter = counter + 1;
            end
            
            if obj.Settings.UseW12
                for i=1:D
                    d(:,:,counter) = ...
                        d(:,:,counter) ...
                            + sqrt(dist_L2_sqr(:,:,i));
                    for j=1:K
                        d(:,:,counter) = ...
                            d(:,:,counter) ...
                                + sqrt(dist_L2_sqr_deriv(:,:,i,j));
                    end
                end
                counter = counter + 1;
            end
            if obj.Settings.UseW1inf
                for i=1:D
                    d(:,:,counter) = ...
                        d(:,:,counter) ...
                            + dist_Linf_sqr(:,:,i);
                    for j=1:K
                        d(:,:,counter) = ...
                            d(:,:,counter) ...
                                + dist_Linf_sqr_deriv(:,:,i,j);
                    end
                end
                counter = counter + 1;
            end
            if obj.Settings.UseW12Prime
                for i=1:D
                    d(:,:,counter) = ...
                        d(:,:,counter) ...
                            + dist_L2_sqr(:,:,i);
                    for j=1:K
                        d(:,:,counter) = ...
                            d(:,:,counter) ...
                                + dist_L2_sqr_deriv(:,:,i,j);
                    end
                end
                d(:,:,counter) = sqrt(d(:,:,counter));
                counter = counter + 1;
            end
            if obj.Settings.UseW1infPrime
                for i=1:D
                    data = dist_Linf_sqr(:,:,i);
                    for j=1:K
                        data = ...
                            max(data,dist_Linf_sqr_deriv(:,:,i,j));
                    end
                    d(:,:,counter) = ...
                        d(:,:,counter) + data;
                end
                counter = counter + 1;
            end
            
            if isa(S1,'single')
                d = single(d);
            end
            
        end % distance
        
        function data = serialize(obj)
            data.Abbreviation = obj.Abbreviation;
            data.Settings = obj.Settings;
        end
    end
    
    methods (Static)
        function obj = load(data)
            obj = MultiDistanceProvider();    
            obj.Abbreviation = data.Abbreviation;
            obj.Settings = data.Settings;
        end
    end
end

