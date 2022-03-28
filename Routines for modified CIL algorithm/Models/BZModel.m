classdef BZModel < Model
    
    properties
      Devices = [0,1]; 
    end
    
    methods
        function obj = BZModel(N)
            obj.Abbreviation = 'BZ';
            
            if (N~=32) && (N~=64)
                error('The specified spatial grid dimension is not supported!')
            end
            
            obj.Settings.N = N;
            if (N==32)
                obj.Settings.InitialStep = 0.01;
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
            
            obj.Parameters.nu1=2/(35^2);  
            obj.Parameters.nu2=16/(35^2);
            obj.Parameters.A=4.5;
            obj.Parameters.B=6.96;
        end  
        
        function IC = IC(obj,varargin)
            [N,theta] = obj.check_IC_input(varargin);
                       
            par = obj.set_parameters(theta);
            
            IC = zeros(2*obj.Settings.N^2,N);
            delta = 0.1;
            for k=1:N
                IC(1:obj.Settings.N^2,k) = ...
                    par.A + delta*rand(1,obj.Settings.N^2);
                IC((obj.Settings.N^2+1):end,k) = ...
                par.B/par.A ...
                + delta*rand(1,obj.Settings.N^2);
            end
            IC = single(IC);           
        end
        
        function S = simulate(obj,varargin)
            [par,N,IC,observer] = check_simulate_input(obj,varargin);
                              
            if ~obj.Settings.UseROCK2
                if (obj.Settings.N == 32)
                    S = BZPatternGeneratorX32(...
                        obj.Settings.T0,...
                        obj.Settings.T1,...
                        obj.Settings.dT,...
                        obj.Settings.InitialStep,...
                        par,... 
                        IC,... 
                        N,...
                        obj.Devices);
                else
                    S = BZPatternGeneratorX64(...
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
                S = R2BZSolverMEX( ...
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
    end
    
    methods (Static)
        function obj = load(data)
            N = data.Settings.N;
            obj = BZModel(N);
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

