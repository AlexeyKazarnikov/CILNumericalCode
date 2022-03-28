% This class is used for the generation of CIL likelihood when only one
% distance is employed.

classdef DefaultCILBuilder < handle
 
    properties
        Settings = struct();
        CIL = struct();
        Curve = struct();
        Chi2 = struct();
        S = [];
        Y = [];
        mu0 = 0;
        Sigma0 = 0;
        Model = {};
    end
    
    methods
        function obj = DefaultCILBuilder(model)         
            obj.Settings.UseBootstrap = true;
            obj.Settings.UseUniformSpacing = false;

            obj.CIL.N = 500;
            obj.CIL.nens = 1000;
            obj.CIL.M = 12;
            obj.CIL.delta = 1e-3;
            obj.CIL.eps = 1e-6;
            
            obj.Chi2.rngindex1 = 1;
            obj.Chi2.rngindex2 = obj.CIL.M;
            obj.Chi2.thr1 = 5e-3;
            obj.Chi2.thr2 = 5e-3;
            
            obj.Model = model;
        end
        
        function curve = estimate_ecdf_constants(obj,theta0,S)
            if nargin < 3
                s1 = obj.Model.simulate(obj.CIL.N,[],theta0);
                s2 = obj.Model.simulate(obj.CIL.N,[],theta0);
            else
                s1 = S(:,:,1);
                s2 = S(:,:,2);
            end
            dist = obj.Model.distance(s1,s2);   
          
            [R0,b] = ...
                nlEstimateECDFConstants( ...
                    dist(dist > obj.CIL.eps), ...
                    obj.CIL.M, ...
                    obj.Settings.UseUniformSpacing, ...
                    obj.CIL.delta ...
                );
                   
            curve.R0 = R0;
            curve.b = b;
            
            obj.Curve = curve;           
        end
        
        function [c0,x0] = generate_ecdf(obj,theta0,S)
            if nargin < 3
                s1 = obj.Model.simulate(obj.CIL.N,[],theta0);
                s2 = obj.Model.simulate(obj.CIL.N,[],theta0);
            else
                s1 = S(:,:,1);
                s2 = S(:,:,2);
            end
            dist = obj.Model.distance(s1,s2);
            
            [c0,x0] = ... 
                nlCreateECDF( ...
                    dist(dist > obj.CIL.eps), ...
                    obj.CIL.M, ...
                    obj.Settings.UseUniformSpacing, ...
                    obj.Curve.R0, ...
                    obj.Curve.b ...
                );
        end
        
        function S = generate_pattern_data(obj,theta0)
            s1 = obj.Model.simulate(obj.CIL.N,[],theta0);
            
            if (~obj.Settings.UseBootstrap)
                nsim = nlEstimateSimulationNumber(obj.CIL.nens);
                S = zeros(size(s1,1),obj.CIL.N,nsim);
                S(:,:,1)=s1;

            for i=2:nsim   
                S(:,:,i) = obj.Model.simulate(obj.CIL.N,[],theta0);
            end
            else
                S = zeros(size(s1,1),obj.CIL.N,2);
                S(:,:,1)=s1;

                S(:,:,2) = obj.Model.simulate(obj.CIL.N,[],theta0);
            end
            
            obj.S = S;
        end
        
        function Y = generate_distribution(obj,show_figures)
            if nargin < 2
                show_figures = true;
            end

            Y = zeros(obj.CIL.nens,obj.CIL.M+1);
            k=1; %counter variable

            % Optimization fix in the case of bootstrapping
            if (obj.Settings.UseBootstrap)
                s1 = obj.S(:,:,1);
                s2 = obj.S(:,:,2);
                dist_bank = obj.Model.distance(s1,s2);
                nsim = obj.CIL.nens;
            else
                nsim = size(obj.S,3);
            end
            
            if show_figures
                hold on
            end
            
            dist_curves = zeros(obj.CIL.nens,size(obj.S,2)^2);

            for i=1:nsim
                if(k>obj.CIL.nens)
                    break;
                end
    
                for j=i+1:nsim
                    if(k>obj.CIL.nens)
                        break;
                    end

                    if (~obj.Settings.UseBootstrap)
                        s1 = obj.S(:,:,i);
                        s2 = obj.S(:,:,j);
                        dist = obj.Model.distance(s1,s2);
                    else        
                        Sind1 = randi(obj.CIL.N,1,obj.CIL.N);
                        Sind2 = randi(obj.CIL.N,1,obj.CIL.N);          

                        dist = dist_bank(Sind1,Sind2);           
                    end
                    
                    dist_curves(k,:) = dist(:);

                    k=k+1; %increasing counter variable value
                end
            end
            
%             [R0,b] = ...
%                 nlEstimateECDFConstantsBatch( ...
%                     dist_curves, ...
%                     obj.CIL.M, ...
%                     obj.Settings.UseUniformSpacing, ...
%                     obj.CIL.delta ...
%                 );
%             
%             curve.R0 = R0;
%             curve.b = b;
%             
%             obj.Curve = curve;
            
            for k=1:size(dist_curves,1)      
                    [c0,x0] = nlCreateECDF( ...
                        dist_curves(k,:), ...
                        obj.CIL.M, ...
                        obj.Settings.UseUniformSpacing, ...
                        obj.Curve.R0, ...
                        obj.Curve.b ...
                        );

                    % Plotting the results (if specified in script settings)
                    if (show_figures)
                        plot(x0,c0,'o-','color','b');
                        drawnow;
                        title(sprintf('k=%i',k))
                    end

                    % Saving values of log-log (empcdf) curve
                    Y(k,:)=c0;
            end
            
            if show_figures
                hold off
            end
            
            obj.Y = Y;
        end
        
        function  [khi_n,x] = perform_chi2_test(obj,show_figures,num_bars)
            if nargin < 2
                show_figures = true;
            end
            
            if nargin < 3
                num_bars = 25;
            end
            
            Yl = obj.Y;
            
            if obj.Chi2.thr1>0 || obj.Chi2.thr2>0
                [obj.Chi2.rngindex1,obj.Chi2.rngindex2] = ...
                    nlGetThresholdNew(Yl);
            else
                [obj.Chi2.rngindex1,obj.Chi2.rngindex2] = ...
                    nlGetThresholdNew(Yl);
            end
            
            Yl=Yl(:,obj.Chi2.rngindex1:obj.Chi2.rngindex2);

            mu = mean(Yl);
            Y0 = Yl-repmat(mu,size(Yl,1),1);
            C=cov(Yl);
            khi2= sum(Y0'.*(C\Y0')); 
            [khi,x]=hist(khi2,num_bars);
            khi_n = khi/sum(khi)/(x(2)-x(1));
            
            if show_figures
                hold on
                bar(x,khi_n)
                plot(x,chi2pdf(x,size(Yl,2)),'r','linewidth',3)
                hold off  
            end
        end
        
        function [mu0, Sigma0] = estimate_distribution_parameters(obj)
            Yl = obj.Y;
            Yl=Yl(:,obj.Chi2.rngindex1:obj.Chi2.rngindex2);
            mu0 = mean(Yl);
            Sigma0 = cov(Yl);
            obj.mu0 = mu0;
            obj.Sigma0 = Sigma0;
        end
        
        function data = serialize(obj)
            data.Type = 'Default';
            data.Settings = obj.Settings;
            data.CIL = obj.CIL;
            data.Curve = obj.Curve;
            data.Chi2 = obj.Chi2;
            data.S = obj.S;
            data.Y = obj.Y;
            data.mu0 = obj.mu0;
            data.Sigma0 = obj.Sigma0;
            data.Model = obj.Model.serialize();
        end
        
        function [] = restore(obj,data)
        end
    end
end

