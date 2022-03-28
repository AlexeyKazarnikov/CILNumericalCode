% This class is used for the generation of CIL likelihood when multiple
% distances are employed.

classdef MultiCILBuilder < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        DistanceProviders = {};
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
        function obj = MultiCILBuilder(model,distance_providers)
            obj.DistanceProviders = distance_providers;
            
            obj.Settings.UseBootstrap = true;
            obj.Settings.UseUniformSpacing = false;

            obj.CIL.N = 500;
            obj.CIL.nens = 1000;
            obj.CIL.M = 12;
            obj.CIL.delta = ...
                1e-3 * ones(1,length(obj.DistanceProviders));
            obj.CIL.eps = 1e-6;
            
            obj.Chi2.rngindex1 = ...
                1 * ones(1,length(obj.DistanceProviders));
            obj.Chi2.rngindex2 = ...
                obj.CIL.M * ones(1,length(obj.DistanceProviders));
            obj.Chi2.thr1 = ...
                5e-3 * ones(1,length(obj.DistanceProviders));
            obj.Chi2.thr2 = ...
                5e-3 * ones(1,length(obj.DistanceProviders));
            
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
            
            curve.R0 = zeros(1,length(obj.DistanceProviders));
            curve.b = zeros(1,length(obj.DistanceProviders));
            
            for nf=1:length(obj.DistanceProviders)
                dist = obj.DistanceProviders{nf}.distance(s1,s2,obj.Model);   

                [curve.R0(nf),curve.b(nf)] = ...
                    nlEstimateECDFConstants( ...
                        dist(dist > obj.CIL.eps), ...
                        obj.CIL.M, ...
                        obj.Settings.UseUniformSpacing, ...
                        obj.CIL.delta(nf) ...
                    );
            end
            
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
            
            c0 = cell(1,length(obj.DistanceProviders));
            x0 = cell(1,length(obj.DistanceProviders));
            
            for nf=1:length(obj.DistanceProviders)
                dist = obj.DistanceProviders{nf}.distance(s1,s2,obj.Model);
            
                [c0{nf},x0{nf}] = ... 
                    nlCreateECDF( ...
                        dist(dist > obj.CIL.eps), ...
                        obj.CIL.M, ...
                        obj.Settings.UseUniformSpacing, ...
                        obj.Curve.R0(nf), ...
                        obj.Curve.b(nf) ...
                    );
            end
        end
        
        function S = generate_pattern_data(obj,theta0)
            s1 = obj.Model.simulate(obj.CIL.N,[],theta0);
            
            if (~obj.Settings.UseBootstrap)
                nsim = nlEstimateSimulationNumber(obj.CIL.nens);
                S = zeros(size(s1,1),Npatterns,nsim);
                S(:,:,1)=s1;

            for i=2:nsim   
                tic
                S(:,:,i) = obj.Model.simulate(obj.CIL.N,[],theta0);
                toc
            end
            else
                S = zeros(size(s1,1),obj.CIL.N,2);
                S(:,:,1)=s1;

                tic
                S(:,:,2) = obj.Model.simulate(obj.CIL.N,[],theta0);
                toc
            end
            
            obj.S = S;
        end
        
        function Y = generate_distribution(obj,show_figures)
            if nargin < 2
                show_figures = true;
            end

            Y = zeros( ...
                obj.CIL.nens, ...
                obj.CIL.M+1, ...
                length(obj.DistanceProviders) ...
                );
            
            k=1; %counter variable

            % Optimization fix in the case of bootstrapping
            if (obj.Settings.UseBootstrap)
                Sgen = [obj.S(:,:,1) obj.S(:,:,2)];
                dist_bank = zeros( ...
                    size(Sgen,2), ...
                    size(Sgen,2), ...
                    length(obj.DistanceProviders) ...
                    );
                
                for nf=1:length(obj.DistanceProviders)
                    dist_bank(:,:,nf) = ...
                        obj.DistanceProviders{nf}.distance(Sgen,Sgen,obj.Model);
                end
                nsim = obj.CIL.nens;
            else
                nsim = size(obj.S,3);
            end
            
            if show_figures
                for nf=1:length(obj.DistanceProviders)
                    subplot(1,length(obj.DistanceProviders),nf);
                    hold on
                end
            end

            for i=1:nsim
                if(k>obj.CIL.nens)
                    break;
                end
    
                for j=i+1:nsim
                    if(k>obj.CIL.nens)
                        break;
                    end
                    
                    for nf=1:length(obj.DistanceProviders)
                        if (~obj.Settings.UseBootstrap)
                            s1 = obj.S(:,:,i);
                            s2 = obj.S(:,:,j);
                            dist = ...
                                obj.DistanceProviders{nf}.distance(s1,s2,obj.Model);
                        else        
                            Sind = randperm(2*obj.CIL.N);
                            Sind1 = Sind(1:obj.CIL.N);
                            Sind2 = Sind(obj.CIL.N+1:end);

                            Sind3 = randi(obj.CIL.N,1,obj.CIL.N);
                            Sind4 = randi(obj.CIL.N,1,obj.CIL.N);
                            dist = dist_bank(Sind1(Sind3),Sind2(Sind4),nf);           
                        end
                        
                        [c0,x0] = nlCreateECDF( ...
                            dist, ...
                            obj.CIL.M, ...
                            obj.Settings.UseUniformSpacing, ...
                            obj.Curve.R0(nf), ...
                            obj.Curve.b(nf) ...
                            );
                    
                        % Plotting the results (if specified in script settings)
                        if (show_figures)
                            subplot(1,length(obj.DistanceProviders),nf);
                            plot(x0,c0,'o-','color','b');
                            title(sprintf('k=%i',k))
                        end

                        % Saving values of log-log (empcdf) curve
                        Y(k,:,nf)=c0;
                    end

                    k=k+1; %increasing counter variable value
                end
            end
            
            if show_figures
                for nf=1:length(obj.DistanceProviders)
                    subplot(1,length(obj.DistanceProviders),nf);
                    hold off
                end
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
            
            for nf=1:length(obj.DistanceProviders)
                Yl = obj.Y(:,:,nf);
                [obj.Chi2.rngindex1(nf),obj.Chi2.rngindex2(nf)] = ...
                    nlGetThresholdNew(Yl);
            end
            
            nel = sum(obj.Chi2.rngindex2 - obj.Chi2.rngindex1 + 1);
            Yf = zeros(obj.CIL.nens,nel);

            k = 1;
            for nf = 1:length(obj.DistanceProviders)
                Yl = obj.Y(:,:,nf);
                Yl=Yl(:,obj.Chi2.rngindex1(nf):obj.Chi2.rngindex2(nf));
                ind1 = k; 
                ind2 = k + size(Yl,2) - 1;
                Yf(:,ind1:ind2) = Yl;
                k = ind2 + 1;
            end

            mu = mean(Yf);
            Y0 = Yf-repmat(mu,size(Yf,1),1);
            C=cov(Yf);
            khi2= sum(Y0'.*(C\Y0')); 
            [khi,x]=hist(khi2,num_bars);
            khi_n = khi/sum(khi)/(x(2)-x(1));
            
            if show_figures
                hold on
                bar(x,khi_n)
                plot(x,chi2pdf(x,size(Yf,2)),'r','linewidth',3)
                hold off  
            end
        end
        
        function [mu0, Sigma0] = estimate_distribution_parameters(obj)
            nel = sum(obj.Chi2.rngindex2 - obj.Chi2.rngindex1 + 1);
            Yf = zeros(obj.CIL.nens,nel);

            k = 1;
            for nf = 1:length(obj.DistanceProviders)
                Yl = obj.Y(:,:,nf);
                Yl=Yl(:,obj.Chi2.rngindex1(nf):obj.Chi2.rngindex2(nf));
                ind1 = k; 
                ind2 = k + size(Yl,2) - 1;
                Yf(:,ind1:ind2) = Yl;
                k = ind2 + 1;
            end

            mu0 = mean(Yf);
            Sigma0 = cov(Yf);
            obj.mu0 = mu0;
            obj.Sigma0 = Sigma0;
        end
        
        function data = serialize(obj)
            data.DistanceProviders = {};
            for k=1:length(obj.DistanceProviders)
                data.DistanceProviders{k} = ...
                    obj.DistanceProviders{k}.serialize();                
            end
            
            data.Type = 'MultiCIL';
            data.Settings = obj.Settings;
            data.CIL = obj.CIL;
            data.Curve = obj.Curve;
            data.Chi2 = obj.Chi2;
            data.S = obj.S;
            data.Y = obj.Y;
            data.mu0 = obj.mu0;
            data.Sigma0 = obj.Sigma0;
            data.Model = ...
                obj.Model.serialize();
        end
        
        function [] = restore(obj,data)
        end
    end
end

