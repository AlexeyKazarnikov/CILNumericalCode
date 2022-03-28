% This abstract class contains the definitions for properties and methods
% used by all models inside this library. In addition to
% abstract methods (which must be implemented in the derived classes), this
% base class contains helper methods (for argument validation, etc).

classdef (Abstract) Model < handle
    
    properties
        % this field contains model-specific settings for numerical solver
        Settings = struct(); 
        
        % this field contains the default values of model parameters
        Parameters = {};
        
        % this field contains the observation operator, which is applied to
        % simulated patterns
        Observer = IdentityObserver();
        
        % this field contains the default distance provider, which is used 
        % to compute distances between pattern data 
        DistanceProvider = LpDistance(2);
        
        % each model must possess the unique abbreviation, which is
        % used during serialization. This value must be adjusted by the
        % derived classes
        Abbreviation = 'Model';
    end
    
    % The following abstract methods must be implemented by all classes
    % which inherit from class Model
    % NOTE! While emplementing these methods for any concrete model please
    % use helper methods (provided below) to check the input data in the
    % uniform way.
    methods (Abstract)
        % This method generates patterns for specified values of 
        % control parameters (theta) and initial conditions (IC).
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
        g = visualize(obj,S,k,dim);
        
        %   This method returns the dimension of the model (number of 
        %   components in the numerical solution).
        %   OUTPUT
        %   N : model dimension
        N = dim(obj);
        
        % This method returns the spatial size of the model output
        %   OUTPUT
        %   S : model output size
        S = size(obj);
        
        % This method returns the spatial steps in the discretized grid
        h = dx(obj);     
    end
    
    methods (Static)
        % This method de-serializes the object of the same class
        %   INPUT
        %   data : structure, which contains the serialized object
        %   OUTPUT
        %   obj : an instance of the respective class
        function obj = load(data)
            error('It is impossible to load an abstract class! Use Model.restore() instead.');
        end
        
        % This method de-serializes the object by recognising the
        % abbreviation and calling the respective method of the derived
        % class
        %   INPUT
        %   data : structure, which contains the serialized object
        %   OUTPUT
        %   obj : an instance of the respective class
        function obj = restore(data)
            if strcmp(data.Abbreviation,'BZ')
                obj = BZModel.load(data);
            elseif strcmp(data.Abbreviation,'GM')
                obj = GMModel.load(data);
            elseif strcmp(data.Abbreviation,'FHN')
                obj = FHNModel.load(data);
            elseif strcmp(data.Abbreviation,'COV')
                obj = CovModel.load(data);
            elseif strcmp(data.Abbreviation,'TriangleModel')
                obj = TriangleModel.load(data);
            elseif strcmp(data.Abbreviation,'MCModel')
                obj = MCModel.load(data);
            elseif strcmp(data.Abbreviation,'MAModel')
                obj = MAModel.load(data);
            elseif strcmp(data.Abbreviation,'MIModel')
                obj = MIModel.load(data);
            elseif strcmp(data.Abbreviation,'CHModel')
                obj = CHModel.load(data);
            elseif strcmp(data.Abbreviation,'CHModelBasic')
                obj = CHModelBasic.load(data);
            elseif strcmp(data.Abbreviation,'HMT')
                obj = HMTModel.load(data);
            elseif strcmp(data.Abbreviation,'HMTBasic')
                obj = HMTModelBasic.load(data);
            else
                error('Model class was not recognised!');
            end
        end
    end
    
    % The following helper methods implement several general routines,
    % which are used in derived classes
    methods              
        % this helper method is used to verify the input arguments of the
        % 'IC' method
        %   INPUT
        %   vararg : cell array, which contains the input arguments
        %   OUTPUT
        %   N : required number of initial conditions vectors
        %   theta : structure with control parameter values as fields
        %   par : structure, which contains all model parameter values as
        %   fields
        function [N,theta,par] = check_IC_input(obj,vararg)
            % field extracting
            N = nlExtractArg(vararg,1);
            theta = nlExtractArg(vararg,2);
            
            % data type checking
            if isempty(N)
                error('Input parameter N must be set!');
            end
            if ~isempty(N) && ~isscalar(N)
                error('Input parameter N must be a scalar value!');
            end
            if ~isempty(theta) && ~isstruct(theta)
                error('Input parameter theta must be a struct!');
            end
            
            % determining values
            if (N <= 0)
                error('Input parameter N must be a positive number!');
            end  
            
            par = obj.set_parameters(theta);            
        end
        
        % this helper method is used to verify the input arguments of the
        % 'simulate' method
        %   INPUT
        %   vararg : cell array, which contains the input arguments
        %   OUTPUT
        %   par : structure, which contains all model parameter values as
        %   fields
        %   N : number of simulations to be performed
        %   IC : matrix of initial conditions (stored in row-major format)
        %   observer : observational operator, which is applied to
        %   simulated patterns
        function [par,N,IC,observer] = check_simulate_input(obj,vararg)
            % field extracting
            N = nlExtractArg(vararg,1);
            IC = nlExtractArg(vararg,2);
            theta = nlExtractArg(vararg,3);
            observer = nlExtractArg(vararg,4);
            
            % data type checking
            if isstruct(N) || (~isempty(N) && ~isscalar(N))
                error('Input parameter N must be a scalar value!');
            end
            if ~isempty(IC) && (~ismatrix(IC) || ndims(IC) ~= 2)
                error('Input parameter IC must be a 2D matrix!');
            end
            if ~isempty(theta) && ~isstruct(theta)
                error('Input parameter theta must be a struct!');
            end
            if ~isempty(observer) && ~isa(observer, 'Observer')
                error('Input parameter observer must be an instance of Observer class!');
            end
            
            % determining values
            par = obj.set_parameters(theta);
            if isempty(N) && isempty(IC)
                error('One of input parameters N and IC must be set!');
            end
            
            if ~isempty(N) && ~isempty(IC) && N ~= size(IC,2)
                error('When both N and IC are provided, they should correspond to each other!');
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
        
        % this helper method is used to verify the input arguments of the
        % 'visualize' method
        %   INPUT
        %   vararg : cell array, which contains the input arguments
        %   OUTPUT
        %   S : matrix, containing simulated patterns in row-major format
        %   k : index of the solution to be plotted
        %   dim : dimension of pattern to plot
        function [S,k,dim] = check_visualize_input(obj,vararg)
            S = nlExtractArg(vararg,1);
            k = nlExtractArg(vararg,2);
            dim = nlExtractArg(vararg,3);
            
            % data type checking
            if isempty(S)
                error('Input parameter S must be set!');
            end
            if ~ismatrix(S) || ndims(S)~=2
                error('Input parameter S must be a 2D matrix!');
            end
            if ~isempty(k) && ~isscalar(k)
                error('Input parameter k must be a scalar value!');
            end
            if ~isempty(dim) && ~isscalar(dim)
                error('Input parameter dim must be a scalar value!');
            end
            if isempty(k)
                k = 1;
            end
            if isempty(dim)
                dim = 1;
            end
            
            if k < 0 || k > size(S,2)
                error('Index k should be in proper range!');
            end
            
            if dim < 0 || dim > obj.dim()
                error('Input parameter dim must be in proper range!');
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
        
        % This method applies the values of control parameters to the
        % vector of model parameters and returns the result. The values
        % inside 'obj.Parameters' field do not change
        %   INPUT
        %   theta : structure with control parameter values as fields
        %   OUTPUT
        %   par : output structure with updated parameter values
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
        
        % This method selects the requested dimensions from the pattern
        % data (generated by the same model)
        %   INPUT
        %   S : pattern data (2D array)
        %   dims : dimensions to be selected (1D array)
        %   OUTPUT
        %   Sd : respective dimensions of the pattern data
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
            
            if isa(S,'single')
                Sd = single(Sd);
            end
        end
        
        %   This method computes the distance matrix between the pattern sets
        %   S1 and S2
        %   INPUT
        %   S1 : first set of patterns, stored in row-major format
        %   S2 : second set of patterns, stored in row-major format
        %   OUTPUT
        %   d : distance matrix (d_ij = dist(S1(i),S2(j)))
        function d = distance(obj,S1,S2)
            if isa(S1,'single') || isa(S2,'single')
                S1 = single(S1);
                S2 = single(S2);
            end
            
            d = obj.DistanceProvider.distance(S1,S2,obj);
        end
        
        % This method serializes the object and saves the result into the
        % output structure
        %   OUTPUT
        %   data : structure, which contains the serialized object
        function data = serialize(obj)
            error('Not implemented yet!');
        end
        
        % This method differentiates the pattern data with respect to all
        % spatial variables
        %   INPUT
        %   S : pattern data (2D array)
        %   OUTPUT
        %   dS : cell array, which contains the differentiated patterns
        function dS = differentiate(obj,S)
            dS = nlDiffPatterns(S,obj);
        end
    end
end

