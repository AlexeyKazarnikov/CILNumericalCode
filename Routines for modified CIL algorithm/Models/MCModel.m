classdef MCModel < Model
    
    properties
        Devices = [0 1];
    end
    
    methods
        function obj = MCModel()
            obj.Abbreviation = 'MCModel';
            
            obj.Parameters.alpha = 1;
            obj.Parameters.beta = 10;
            obj.Parameters.D = 0.75;
            obj.Parameters.delta = 10.0;
            obj.Parameters.L = 10;
            obj.Parameters.N = 64;
            obj.Parameters.tau = 200;
            
%             obj.Settings.sol_par.dt = 1e-4;
%             obj.Settings.sol_par.omega = 0.25;
%             obj.Settings.sol_par.abs_tol = 5e-4;
%             obj.Settings.sol_par.max_iter = 50;
%             obj.Settings.sol_par.max_step = 100;
%             obj.Settings.sol_par.conv_norm = 1e-7;
%             obj.Settings.sol_par.jac_step = 1e-8;
%             obj.Settings.sol_par.max_time = 1000;
%             obj.Settings.sol_par.max_step_number = 500;
            
            obj.Settings.sol_par.dt = 1e-3;
            obj.Settings.sol_par.omega = 1;
            obj.Settings.sol_par.abs_tol = 1e-4;
            obj.Settings.sol_par.rel_tol = 1e-4;
            obj.Settings.sol_par.max_iter = 100;
            obj.Settings.sol_par.max_step = 500;
            obj.Settings.sol_par.conv_norm = 1e-8;
            obj.Settings.sol_par.jac_step = 1e-8;
            obj.Settings.sol_par.max_time = 100;
            obj.Settings.sol_par.max_step_number = 1000;
            obj.Settings.sol_par.jac_update_interval = 50;

        end 
        
        function IC = IC(obj,varargin)
            [N,theta,par] = obj.check_IC_input(varargin);
            IC = zeros(2*obj.Parameters.N+1,N);
            
            for k=1:N
                IC(1:obj.Parameters.N,k) = ...
                    1 + 0.01*rand(1,obj.Parameters.N);
                IC((obj.Parameters.N+1):end-1,k) = ...
                    abs(0.001 * (1 + sqrt(0.2) * randn(1,obj.Parameters.N)));

%                   xr = linspace(0,par.L,par.N);
%                   
%                   u0 = 1;         
%                   for r = 1:8
%                       cr = 0.001*randn();
%                       u0 = u0 + cr*cos(pi*r/par.L*xr);
%                   end
%                   
%                   IC(1:obj.Parameters.N,k) = u0;
            end          
        end
        
        function S = simulate(obj,varargin)
            [par,N,IC,observer] = check_simulate_input(obj,varargin);
            
            devices = obj.Devices;
            
            if N<=100
                devices = devices(1);    
            end
            
            Nsim = N;
            
            S = MCImplicitSolverMEX(...
                    par, ...
                    obj.Settings.sol_par, ...
                    Nsim, ...
                    IC, ...
                    devices ...
                );
            
            % deleting lagrange multiplier component
            S(end,:) = [];
            
            % removing curvature shift
            S(1:par.N,:) = ...
               S(1:par.N,:) - min(S(1:par.N,:),[],1);
            
            S = observer.transform(S,obj);
        end
        
        function g = visualize(obj,varargin)
            [S,k,dim] = obj.check_visualize_input(varargin);
            
            x = linspace(0,1,obj.Parameters.N);
            
            hold on
            plot(x,S(1:obj.Parameters.N,k),'r')
            if size(S,1) == 2*obj.Parameters.N + 1
                plot(x,S(obj.Parameters.N+1:end-1,k),'b')
            else
                plot(x,S(obj.Parameters.N+1:end,k),'b')
            end
            hold off
        end
        
        function N = dim(obj)
            N = 2;
        end
        
        function S = size(obj)
            S = obj.Parameters.N;
        end
        
        function h = dx(obj)
            h = obj.Parameters.L / (obj.Parameters.N - 1);
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
            obj = MCModel();
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

