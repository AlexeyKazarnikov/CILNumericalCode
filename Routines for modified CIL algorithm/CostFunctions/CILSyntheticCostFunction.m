classdef CILSyntheticCostFunction < CostFunction
    
    properties

    end
    
    methods
        function obj = CILSyntheticCostFunction()
            obj.Abbreviation = 'CILSynthetic';
            
            % total number of patterns in the training set
            obj.Settings.M = 1000;
            
            % number of distances used to form one eCDF curve
            obj.Settings.N = 1000;
            
            % number of eCDF curves in the training set
            obj.Settings.Ntr = 1000;
            
            % using the uniform placing of bins
            obj.Settings.UseUniformBins = false;
            
            % number of radii
            obj.Settings.Nr = 18;
            
            % thresholds
            obj.Settings.thr1 = 5e-3;
            obj.Settings.thr2 = 5e-3;
        end
        
        function [res,data] = evaluate(obj,varargin) 
            [model,theta,sref] = check_evaluate_input(obj,varargin);
            
            % simulating pattern data
            S = model.simulate(obj.Settings.M,[],theta);
            
            % early rejection test
            for k = 1:model.dim()
                Sk = model.select_dimensions(S,k);
                idxk = max(abs(Sk - repmat(mean(Sk,1),size(Sk,1),1))) < 1e-3;
                if sum(idxk) > 0
                    res = 1e15;
                    data = [];
                    return;
                end
            end

            % computing distances between the respective patterns
            D0 = model.distance(S,[]);
            
            % estimating CIL constants for all used distances
            dist_number = size(D0,3);
            R0 = zeros(1,dist_number);
            base = zeros(1,dist_number);
            for d=1:dist_number
                % selecting distances for current norm
                D0_est = D0(:,:,d);
                
                % removing diagonal elements (zeros)   
                D0_est(logical(eye(size(D0_est)))) = [];
                
                % estimating eCDF constants
%                 [R0(d),base(d)] = nlEstimateECDFConstants(...
%                     D0_est,...
%                     obj.Settings.Nr,...
%                     obj.Settings.UseUniformBins...
%                     );
                
                [R0(d),base(d)] = nlEstimateECDFConstantsNew(...
                    D0_est,...
                    obj.Settings.Nr,...
                    obj.Settings.UseUniformBins,...
                    1e-4,...
                    obj.Settings.N,...
                    size(sref,2),...
                    100 ...
                    );
            end
                           
            % CIL curves generation 
            Y = zeros(obj.Settings.Nr+1,obj.Settings.Ntr,dist_number);
            rngindex1 = zeros(1,dist_number);
            rngindex2 = zeros(1,dist_number);
            ref_pattern_number = size(sref,2);

            for d=1:dist_number
                for k = 1:obj.Settings.Ntr      
                    % selecting random indexes for reference patterns
                    %tic
                    n0 = randi(obj.Settings.M,1,ref_pattern_number);
                    %mt1 = mt1 + toc;

                    % selecting M random indexes to form eCDF curve
                    % here we use the range [1, M-1] to remove the indexes
                    % equal to n0 by shifting.  
                    %tic
                    max_ind = obj.Settings.M - ref_pattern_number;
                    nind = randi(max_ind,1,obj.Settings.N);
                    ref_ind = 1:obj.Settings.M;
                    ref_ind(n0) = [];
                    nind = ref_ind(nind);
                    %mt2 = mt2 + toc;

                    % creating eCDF curve                  
                    %tic
                    [c0,~,~]=nlCreateECDF(...
                        D0(n0,nind,d),...
                        obj.Settings.Nr,...
                        obj.Settings.UseUniformBins,...
                        R0(d),...
                        base(d)...
                        );
                    %mt4 = mt4 + toc;
                    
                    Y(:,k,d) = c0;
                end
            end

            for d=1:dist_number
                [rngindex1(d),rngindex2(d)] = nlGetThresholdNew(Y(:,:,d)');
                
                % bad distribution test
                if (rngindex1(d) >= rngindex2(d))
                    res = 1e15;
                    data = [];
                    return;
                end
            end

            % function evaluation (chi-squared test)   
            nel = sum(rngindex2 - rngindex1 + 1);
            Yf = zeros(obj.Settings.Ntr,nel);

            k = 1;
            for nf = 1:dist_number
                Yl = Y(:,:,nf);
                Yl=Yl(rngindex1(nf):rngindex2(nf),:)';
                ind1 = k; 
                ind2 = k + size(Yl,2) - 1;
                Yf(:,ind1:ind2) = Yl;
                k = ind2 + 1;
            end
                   
            mu0 = mean(Yf);
            C=cov(Yf);

            % reference curve generation
            if obj.Settings.N <= obj.Settings.M
                nref = randperm(obj.Settings.M);
                nref = nref(1:obj.Settings.N);
                D0_ref = model.distance(sref,S(:,nref));
            else
                D0_ref = model.distance(sref,S);
                nref = randi(obj.Settings.M,1,obj.Settings.N);
                D0_ref = D0_ref(:,nref,:);
            end

            Yjoint = [];
            for nf=1:dist_number
                D = D0_ref(:,:,nf);      
                [Yref,~,~]=...
                  nlCreateECDF( ...
                  D, ...
                  obj.Settings.Nr, ...
                  obj.Settings.UseUniformBins, ...
                  R0(nf), ...
                  base(nf));
                Yref = Yref(rngindex1(nf):rngindex2(nf));
                Yjoint = [Yjoint Yref];
            end

            res = (Yjoint-mu0)*(C\(Yjoint-mu0)');

            % steady-state 'bug' fix
            res = abs(res);

            if (isinf(res) || isnan(res))
                res = 1e15;
            end

            data.Y = Y;
            data.c0_ref = Yjoint;
            data.rngindex1 = rngindex1;
            data.rngindex2 = rngindex2;
            data.R0 = R0;
            data.base = base;
            data.UseUniformBins = obj.Settings.UseUniformBins;
        end
        
        function f = visualize(obj,data)
            if isempty(data)
                return
            end
            
            D = size(data.Y,3);
            
            ref_start = 1;
            
            if D > 1
                for d=1:D                   
                    rngval = data.rngindex1(d):data.rngindex2(d);
                    rng_len = length(rngval);
                                 
                    subplot(1,D,d);
                    hold on
                    plot(data.Y(:,:,d),'bo-');
                    plot( ...
                        rngval,data.c0_ref(ref_start:ref_start + rng_len - 1), ...
                        'ro-',...
                        'linewidth',3 ...
                        );
                    hold off
                    
                    ref_start = ref_start + rng_len;
                end
            end
        end
        
        function data = serialize(obj)
            data.Abbreviation = obj.Abbreviation;
            data.Settings = obj.Settings;
        end
    end
    
    methods (Static)
        function obj = load(data)
            obj = CILSyntheticCostFunction();
            obj.Abbreviation = data.Abbreviation;
            obj.Settings = data.Settings;
        end
    end
end

