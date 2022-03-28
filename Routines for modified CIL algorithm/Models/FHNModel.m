classdef FHNModel < Model
    
    properties
      Devices = [0,1];  
    end
    
    methods
        function obj = FHNModel(N)
            obj.Abbreviation = 'FHN';
            
            if (N~=32) && (N~=64)
                error('The specified spatial grid dimension is not supported!')
            end
            
            obj.Settings.N = N;
            if (N==32)
                obj.Settings.InitialStep = 0.0045;
            else
                obj.Settings.InitialStep = 0.001;
            end

            obj.Settings.T0 = 0;
            obj.Settings.dT = 201;
            obj.Settings.T1 = 100;
            obj.Settings.UseROCK2 = false;
            
            obj.Settings.ROCK2Settings.initial_time_step = 1e-5;
            obj.Settings.ROCK2Settings.final_time_point = 100;
            obj.Settings.ROCK2Settings.conv_norm = 1e-4;
            obj.Settings.ROCK2Settings.conv_stage_number = 150;
            obj.Settings.ROCK2Settings.abs_tol = 1e-2;
            obj.Settings.ROCK2Settings.rel_tol = 1e-2;
            obj.Settings.ROCK2Settings.grid_resolution = N;
            
            obj.Parameters.nu1=5e-3 / .1;  
            obj.Parameters.nu2=2.8e-4;
            obj.Parameters.mu=1.0;
            obj.Parameters.eps=10.0;
            obj.Parameters.A=0;
            obj.Parameters.B=1;
            obj.Parameters.alpha = 1;
        end
   
        
        function IC = IC(obj,varargin)
            [N,~] = obj.check_IC_input(varargin);
            
            delta = 0.1;
            
            IC = zeros(2 * obj.Settings.N^2, N);
            
            for k=1:N
                IC(1:obj.Settings.N^2,k) = ...
                    delta*rand(1,[obj.Settings.N]^2);
                IC((obj.Settings.N^2+1):end,k) = ...
                    delta*rand(1,obj.Settings.N^2);
            end
            IC = single(IC);           
        end
        
        function S = simulate(obj,varargin)   
            [par,N,IC,observer] = check_simulate_input(obj,varargin);                       
            
            if ~obj.Settings.UseROCK2
                if (obj.Settings.N == 32)
                    S = FHNPatternGeneratorX32(...
                        obj.Settings.T0,...
                        obj.Settings.T1,...
                        obj.Settings.dT,...
                        obj.Settings.InitialStep,...
                        par,... 
                        IC,... 
                        N,...
                        obj.Devices);
                else
                    S = FHNPatternGeneratorX64(...
                        obj.Settings.T0,...
                        obj.Settings.T1,...
                        obj.Settings.dT,...
                        obj.Settings.InitialStep,...
                        par,... 
                        IC,... 
                        N,...
                        obj.Devices);
                end
            else
                S = R2FHNSolverMEX( ...
                    par, ...
                    obj.Settings.ROCK2Settings, ...
                    N, ...
                    double(IC), ...
                    obj.Devices ...
                    );
            end
                      
            S = observer.transform(S,obj);
        end
        
        function g = visualize(obj,varargin)
            [S,k,dim] = obj.check_visualize_input(varargin);
            
            data = obj.select_dimensions(S(:,k),dim);
            
            data = reshape(data, obj.Settings.N, obj.Settings.N);
            
            h = 1 / (obj.Settings.N - 1);
            x = 0:h:1;
            y = 0:h:1;
            
            g=surf(x,y,data);
            view([0 90])
            set(g,'linestyle','none');
        end
        
        function N = dim(obj)
            N = 2;
        end
        
        function S = size(obj)
            S = [obj.Settings.N obj.Settings.N];
        end
        
        function h = dx(obj)
            h = zeros(1,2);
            h(1) = 1 / (obj.Settings.N - 1);
            h(2) = 1 / (obj.Settings.N - 1);
        end
        
        function data = serialize(obj)         
            data.Settings = obj.Settings; 
            data.Parameters = obj.Parameters;
            data.Observer = obj.Observer.serialize(); 
            data.DistanceProvider = obj.DistanceProvider.serialize();      
            data.Abbreviation = obj.Abbreviation;
            data.Devices = obj.Devices;
        end
        
%         function d = distance(obj,S1,S2)
%             d = (1/(obj.Settings.N-1)) * DistanceMatrixCUDA(double(S1),double(S2));
%         end
    end
    
    methods (Static)
        function obj = load(data)
            N = data.Settings.N;
            obj = FHNModel(N);
            obj.Settings = data.Settings; 
            obj.Parameters = data.Parameters;
            obj.Observer = ...
                Observer.restore(data.Observer); 
            obj.DistanceProvider = ...
                DistanceProvider.restore(data.DistanceProvider);      
            obj.Abbreviation = data.Abbreviation;
            obj.Devices = data.Devices;
        end
    end
end

