function S = RDPatternGenerator(f,g,T0,T1,par,IC,Nsim,N)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
S = zeros(2*N^2,Nsim);

for k = 1:Nsim
    rhs = @(t,y) MOL_function(t,y,f,g,par,N); % anonymous function for ode45
    t = [T0 T1/2 T1]; % time interval
    [~,yr] = ode45(rhs,t,double(IC(:,k))); % obtaining numerical solution
    S(:,k) = single(yr(end,:));
end

end

