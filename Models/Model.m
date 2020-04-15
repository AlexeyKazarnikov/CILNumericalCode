classdef (Abstract) Model < handle
%   This abstract class represents a model, which could be used by other
%   scripts in this library. It 
    
    properties
        % this field contains model-specific settings for numerical solver
        Settings = struct(); 
        
        % this field contains the default values of model parameters
        Parameters = {};
        
        % this field contains the observation operator, which is applied to
        % simulated patterns
        Observer = IdentityObserver(); 
        
        Abbreviation = 'Model';
    end
    
    % The following abstract methods must be implemented by all classes
    % which inherit from class Model
    methods (Abstract)
        % This method returns simulated patterns (produced by model), 
        % obtained for specified values of control parameters (theta) 
        % and initial conditions (IC).
        %   INPUT
        %   N : number of simulations to be performed
        %   IC : matrix of initial conditions (stored in row-major format)
        %   theta : structure with control parameter values as fields, all
        %   other parameter values (not specified in theta) are taken from
        %   the default values of model parameters (obj.Parameters)
        %   observer : observational operator, which is applied to
        %   simulated patterns (could be used for min-max normalization)
        %   OUTPUT
        %   S : matrix, containing output patterns in row-major format
        S = simulate(obj,N,IC,theta,observer);
        
        % this method generates randomized initial conditions to be used in
        % the 'simulate' method.
        %   INPUT
        %   N : required number of initial conditions vectors
        %   theta : structure with control parameter values as fields
        %   OUTPUT
        %   IC : matrix, containing resulting initial conditions in
        %   row-major format
        IC = IC(obj,N,theta); 
        
        % This method plots the specified pattern
        %   INPUT
        %   S : matrix, containing simulated patterns in row-major format
        %   k : index of the solution to be plotted
        %   dim : dimension of pattern to plot
        %   OUTPUT
        %   g : MATLAB graphical object handle
        g = show(obj,S,k,dim);
        
        %   This method returns the dimension of the model (number of 
        %   components in the numerical solution).
        %   OUTPUT
        %   N : model dimension
        N = dim(obj); 
        
        %   This method computes the distance matrix between the pattern sets
        %   S1 and S2
        %   INPUT
        %   S1 : first set of patterns, stored in row-major format
        %   S2 : second set of patterns, stored in row-major format
        %   OUTPUT
        %   d : distance matrix (d_ij = dist(S1(i),S2(j)))
        d = distance(obj,S1,S2);          
    end
    
    % The following non-abstract methods implement several general routines,
    % which are used in inherited classes
    methods
               
        %   this helper method is used to extract arguments from MATLAB 
        %   varargin structure    
        function val = extract_arg(obj,vararg,k)
            if length(vararg) >= k
                val = vararg{k};
            else
                val = [];
            end
        end
        
        % this helper method is used to verify the input arguments of the
        % 'simulate' method
        function [par,N,IC,observer] = check_simulate_input(obj,vararg) 
            N = obj.extract_arg(vararg,1);
            IC = obj.extract_arg(vararg,2);
            theta = obj.extract_arg(vararg,3);
            observer = obj.extract_arg(vararg,4);
            
            par = obj.set_parameters(theta);
            if isempty(N) && isempty(IC)
                error('Input parameter N or input parameter IC has to be set!');
            end
            if isempty(N)
                N = size(IC,2);
            end
            if isempty(IC)
                IC = obj.IC(N,theta);
            end
            if isempty(observer)
                observer = obj.Observer;
            end
        end
		
        % This method performs the min-max normalization of the data.
        %   INPUT
        %   S : matrix, containing resulting solutions as columns
        %   OUTPUT
        %   Sn : normalized matrix
		function Sn = normalize(obj,S)
            D = size(S,1) / obj.dim(); % number of elements in one dimension
            Sn = S;
            
            for k=1:obj.dim()
                ind1 = (k-1)*D+1;
                ind2 = k*D;
                data = Sn(ind1:ind2,:);
                M = max(data,[],1);
                m = min(data,[],1);
                data = (data - repmat(m,size(data,1),1))...
                    ./(repmat(M-m,size(data,1),1));
                Sn(ind1:ind2,:)=data;
            end           			
        end
        
        function par = set_parameters(obj,theta)
            par = obj.Parameters;
            
            if isempty(theta)
                return;
            end
            
            parNames = fieldnames(theta);
            for k=1:length(parNames)
                if isfield(par,parNames{k})
                    parValue = getfield(theta,parNames{k});
                    par = setfield(par,parNames{k},parValue);
                else
                    error('Fields in input parameter structure are incorrect!');
                end
            end
        end
        
        function Sd = select_dimensions(obj,S,dims)
            D = size(S,1) / obj.dim(); % number of elements in one dimension
            Sd = zeros(D*length(dims),size(S,2));
            for k = 1:length(dims)
                % range in input matrix
                ind1 = (dims(k)-1)*D+1;
                ind2 = dims(k)*D;
                Sk = S(ind1:ind2,:);
                
                % range in output matrix
                ind1 = (k-1)*D+1;
                ind2 = k*D;
                Sd(ind1:ind2,:) = Sk;
            end
        end
    end
end

