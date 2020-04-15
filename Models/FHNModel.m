classdef FHNModel < Model
    
    properties
        
    end
    
    methods
        function obj = FHNModel(N)
            if (N~=32) && (N~=64)
                error('The specified spatial grid dimension is not supported!')
            end
            
            obj.Settings.N = N;
            if (N==32)
                obj.Settings.InitialStep = 0.002;
            else
                obj.Settings.InitialStep = 0.001;
            end

            obj.Settings.T0 = 0;
            obj.Settings.dT = 201;
            obj.Settings.T1 = 100;
            obj.Settings.delta = 0.01;
            obj.Settings.devices = 0;
            
            obj.Parameters.nu1=5e-3 / .1;  
            obj.Parameters.nu2=2.8e-4;
            obj.Parameters.mu=1.0;
            obj.Parameters.eps=10.0;
            obj.Parameters.A=0;
            obj.Parameters.B=1;
            
            obj.Abbreviation = 'FHN';
        end
   
        
        function IC = IC(obj,N,theta)

            IC = zeros(2*[obj.Settings.N]^2,N);

            for k=1:N
                IC(1:obj.Settings.N^2,k) = ...
                    obj.Settings.delta*rand(1,[obj.Settings.N]^2);
                IC((obj.Settings.N^2+1):end,k) = ...
                    obj.Settings.delta*rand(1,obj.Settings.N^2);
            end
            
            IC = single(IC);           
        end
        
        function S = simulate(obj,varargin)
            
            [par,N,IC,observer] = ...
                check_simulate_input(obj,varargin);
                       
            S = FHNPatternGenerator(...
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
                S([obj.Settings.N]^2+1:end,k),...
                [obj.Settings.N],...
                [obj.Settings.N]);
            h = 1 / ([obj.Settings.N] - 1);
            x = 0:h:1;
            y = 0:h:1;
            
            %f = figure(fignum);
            %clf
            g=surf(x,y,data);
            %xlabel('x')
            %ylabel('y')
            view([0 90])
            set(g,'linestyle','none');
        end
        
        function N = dim(obj)
            N = 2;
        end
        
        function d = distance(obj,S1,S2)
            d = (1/(obj.Settings.N-1)) * DistanceMatrixMEX(S1,S2);
        end
    end
end

