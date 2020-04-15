classdef BZModel < Model
    
    methods
        function obj = BZModel(N)
            if (N~=32) && (N~=64)
                error('The specified spatial grid dimension is not supported!')
            end
            
            obj.Settings.N = N;
            if (N==32)
                obj.Settings.InitialStep = 0.01;
            else
                obj.Settings.InitialStep = 0.0049;
            end

            obj.Settings.T0 = 0;
            obj.Settings.T1 = 100;
            obj.Settings.dT = 101;
            obj.Settings.delta = 0.01;
            obj.Settings.devices = 0;
            
            obj.Parameters.nu1=2/(35^2);  
            obj.Parameters.nu2=16/(35^2);
            obj.Parameters.A=4.5;
            obj.Parameters.B=8.72;
            
            obj.Abbreviation = 'BZ';
                   
        end
        
        function IC = IC(obj,N,theta)
            par = obj.Parameters;
            if nargin > 2
                par = obj.set_parameters(theta);
            end
            
            IC = zeros(2*obj.Settings.N^2,N);

            for k=1:N
                IC(1:obj.Settings.N^2,k) = ...
                    par.A + obj.Settings.delta*rand(1,[obj.Settings.N]^2);
                IC((obj.Settings.N^2+1):end,k) = ...
                par.B/par.A ...
                + obj.Settings.delta*rand(1,obj.Settings.N^2);
            end
            
            IC = single(IC);           
        end
        
        function S = simulate(obj,varargin)       
            [par,N,IC,observer] = ...
                check_simulate_input(obj,varargin);
                                  
            S = BZPatternGenerator(...
                obj.Settings.T0,...
                obj.Settings.T1,...
                obj.Settings.dT,...
                obj.Settings.InitialStep,...
                par,... 
                IC,... 
                N,...
                obj.Settings.devices);           
                        
            S = observer.transform(S,obj);       
        end
        
        function g = show(obj,S,k)
            if (nargin < 3)
                k = 1;
            end
            
            data = reshape(...
                S(1:[obj.Settings.N]^2,k),...
                [obj.Settings.N],...
                [obj.Settings.N]);
            h = 1 / ([obj.Settings.N] - 1);
            x = 0:h:1;
            y = 0:h:1;
            
            g=surf(x,y,data);
            view([0 90])
            set(g,'linestyle','none');
        end
        
        function N = dim(obj)
            N = 2;
        end
        
        function d = distance(obj,S1,S2)
            d = (1/(obj.Settings.N-1)) * DistanceMatrix(S1,S2);
        end
    end
end

