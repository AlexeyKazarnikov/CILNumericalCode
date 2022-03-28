classdef HMTModelBasic < Model
    
    properties
      Devices = [0,1];  
    end
    
    methods
        function obj = HMTModelBasic(N)
            obj.Abbreviation = 'HMTBasic';
            
            obj.Settings.sol_par.initial_time_step = 1e-5;
            obj.Settings.sol_par.final_time_point = 50;
            obj.Settings.sol_par.conv_norm = 1e-4;
            obj.Settings.sol_par.conv_stage_number = 150;
            obj.Settings.sol_par.abs_tol = 1e-2;
            obj.Settings.sol_par.rel_tol = 1e-2;
            obj.Settings.sol_par.grid_resolution = N; 
            
            % system parameters         
            obj.Parameters.nu1 = 0;  
            obj.Parameters.nu2 = 1;
            obj.Parameters.m1=1.44;
            obj.Parameters.m2=2;
            obj.Parameters.m3=4.1;
            obj.Parameters.k=0.01;        
        end
   
        
        function IC = IC(obj,varargin)
            [N,theta] = obj.check_IC_input(varargin);
            
            par = obj.set_parameters(theta);
            
            Nres = obj.Settings.sol_par.grid_resolution;

            r = rand(20*Nres,2*N);
            for k=1:10
                r = movmean(r,Nres,1);
            end

            r = r(Nres + (1:Nres) * 10,:);
            
            IC = zeros(2*Nres,N);
            IC(1:Nres,:) = 3 + 0.01*r(:,1:N);
            %IC(Nres+1:end,:) = 3 + 0.01*r(:,N+1:end);
            
%             IC = 3 + 1e-5*randn(2*obj.Settings.sol_par.grid_resolution,N);    
%             IC(1:obj.Settings.sol_par.grid_resolution,:) = ...
%                 movmean(IC(1:obj.Settings.sol_par.grid_resolution,:),3,1);
%             IC(1:obj.Settings.sol_par.grid_resolution,:) = ...
%                 movmean(IC(1:obj.Settings.sol_par.grid_resolution,:),9,1);
%             IC(1:obj.Settings.sol_par.grid_resolution,:) = ...
%                 movmean(IC(1:obj.Settings.sol_par.grid_resolution,:),6,1);
%             IC(1:obj.Settings.sol_par.grid_resolution,:) = ...
%                 movmean(IC(1:obj.Settings.sol_par.grid_resolution,:),3,1);
%             IC(1:obj.Settings.sol_par.grid_resolution,:) = ...
%                 movmean(IC(1:obj.Settings.sol_par.grid_resolution,:),3,1);
%             
%             IC(obj.Settings.sol_par.grid_resolution+1:end,:) = ...
%                 movmean(IC(obj.Settings.sol_par.grid_resolution+1:end,:),3,1);
%             IC(obj.Settings.sol_par.grid_resolution+1:end,:) = ...
%                 movmean(IC(obj.Settings.sol_par.grid_resolution+1:end,:),9,1);
%             IC(obj.Settings.sol_par.grid_resolution+1:end,:) = ...
%                 movmean(IC(obj.Settings.sol_par.grid_resolution+1:end,:),6,1);
            
%             xr = linspace(0,1,obj.Settings.sol_par.grid_resolution);
%             
%             for k=1:N
%                 c = 0.001*randn(2,10);
%                 phi = randi(20,2,10);
%             
%                 for j=1:size(c,2)
%                     IC(1:length(xr),k) = ...
%                         IC(1:length(xr),k)' + ...
%                         c(1,j)*cos(pi*phi(1,j)*xr);
%                     IC(length(xr)+1:end,k) = ...
%                         IC(length(xr)+1:end,k)' + ...
%                         c(2,j)*cos(pi*phi(2,j)*xr);
%                 end
%             end
            
%             IC(1:obj.Settings.sol_par.grid_resolution,:) = ...
%                 3 + 1e-5*randn(obj.Settings.sol_par.grid_resolution,N);
            
%             IC(obj.Settings.sol_par.grid_resolution+1:end,:) = ...
%                 repmat((3 + 0.0001 * cos(pi*3*xr))',1,N);
            
            %IC = 3 + 1e-5*randn(2*obj.Settings.sol_par.grid_resolution,N);
            %IC = smoothdata(IC);
        end
        
        function S = simulate(obj,varargin)
            [par,N,IC,observer] = check_simulate_input(obj,varargin); 
            
            devices = obj.Devices;
            if N < 100
                devices = devices(1);
            end
            
            S = R2HMTSolverMEX( ...
                par, ...
                obj.Settings.sol_par, ...
                N, ...
                double(IC), ...
                devices);
            
            S = S(obj.Settings.sol_par.grid_resolution+1:end,:);
                  
            S = observer.transform(S,obj);
            
        end
        
        function g = visualize(obj,varargin)
            [S,k,dim] = obj.check_visualize_input(varargin);
            
            data_v = S(:,k);
               
            h = 1 / (obj.Settings.sol_par.grid_resolution - 1);
            x = 0:h:1;
            g{1}=plot(x,data_v,'b');
        end
        
        function N = dim(obj)
            N = 1;
        end
        
        function S = size(obj)
            S = [obj.Settings.sol_par.grid_resolution];
        end
        
        function h = dx(obj)
            N = obj.Settings.sol_par.grid_resolution;
            h = 1 / (N - 1);
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
            N = data.Settings.sol_par.grid_resolution;
            obj = HMTModelBasic(N);
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

