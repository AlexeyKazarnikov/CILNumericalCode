classdef CDIMAModel < Model
    
    properties
      Devices = [0];  
    end
    
    methods
        function obj = CDIMAModel(N)
            obj.Abbreviation = 'CDIMA';
            
            obj.Settings.N = N;
      
            obj.Settings.initial_time_step = 1e-5;
            obj.Settings.final_time_point = 200000;
            obj.Settings.min_time_point = 30000;
            obj.Settings.conv_norm = 2e-7;
            obj.Settings.abs_tol = 1e-5;
            obj.Settings.rel_tol = 1e-5;
            obj.Settings.grid_resolution = N;
            obj.Settings.max_step_number = 100000;
            obj.Settings.norm_average_window = 1000;
            obj.Settings.BCType = 2;
            obj.Settings.ExperimentalSetup = false;

            obj.Parameters.L = 50;  
            obj.Parameters.d = 1.07;
            obj.Parameters.a = 8.8;
            obj.Parameters.b = 0.09;
            obj.Parameters.sigma = 50;

        end
   
        
        function IC = IC(obj,varargin)
            [N,theta] = obj.check_IC_input(varargin);               
            par = obj.set_parameters(theta);

            v0 = par.a / 5;
            w0 = 1 + v0.^2;
            
            delta = 0.01;
            
            IC = [ ...
                v0 + delta * rand(obj.Settings.N^2, N); ...
                w0 + delta * rand(obj.Settings.N^2, N); ...
                ];
                    
        end
        
        function [S, tend, rhs] = simulate(obj,varargin)  
            [par,N,IC,observer] = check_simulate_input(obj,varargin);
               
            if obj.Settings.BCType == 2
                [S, tend, rhs] = R2CDIMASolverMEX( ...
                    par, ...
                    obj.Settings, ...
                    N, ...
                    double(IC), ...
                    obj.Devices ...
                    );
            elseif obj.Settings.BCType == 4
                error('Not implemented yet!');
%                 S = R2CDIMASolverPMEX( ...
%                     par, ...
%                     obj.Settings, ...
%                     N, ...
%                     double(IC), ...
%                     obj.Devices ...
%                     );
            else
                error('Boundary condition type is not supported!')
            end

            if obj.Settings.ExperimentalSetup
                S = S(1 : obj.Settings.N^2, :);
            end

            S = observer.transform(S,obj);
        end
        
        function g = visualize(obj,varargin)
            [S,k,dim] = obj.check_visualize_input(varargin);
            
            data = obj.select_dimensions(S(:,k),dim);
            
            data = reshape(data, obj.Settings.N, obj.Settings.N);
            
            x = linspace(0, 1, obj.Settings.N);
            y = linspace(0, 1, obj.Settings.N);
 
            g=surf(x,y,data);
            view([0 90])
            set(g,'linestyle','none');

            f = gca;
            colormap(f, flipud(colormap(f)));
        end
        
        function N = dim(obj)
            if ~obj.Settings.ExperimentalSetup
                N = 2;
            else
                N = 1;
            end
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
            obj = CDIMAModel(N);
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

