function [ result ] = MOL_function( t,y,f,g,par,N)
%MOL_function evaluates the MOL discretization of RD system for specified 
% state vector y and time t
%   INPUT
%   t: current time point (scalar)
%   y: system state vector (column vector)
%   f: function handle for first reaction term
%   g: function handle for second reaction term
%   p: parameter vector (struct)
%   grid: discretization settings (struct)
%
%   OUTPUT
%   result: computed phase vector (column vector)

% unpacking data
hx = 1/(N+1);
hy = 1/(N+1);

nu1 = par.nu1;
nu2 = par.nu2;

dim = N*N;

% switching from vector to matrix representation
U = reshape(y(1:dim),N,N)';
V = reshape(y(dim+1:end),N,N)';

% MATLAB is extremely efficient in using matrix computations. Therefore we
% avoid using cycles for computing the result. Instead we formulate the
% task in matrix form

% defining the neighbours for all u_ij and v_ij (taking into account
% Neumann boundary conditions)

% first component
Ur = [U(:,2:end) U(:,end)]; % right (u(i,j)->u(i,j+1),u(i,N) -> u(i,N))
Ul = [U(:,1) U(:,1:end-1)]; % left (u(i,j)->u(i,j-1),u(i,1) -> u(i,1))
Ut = [U(1,:); U(1:end-1,:)]; % top (u(i,j)->u(i-1,j),u(1,j) -> u(1,j))
Ub = [U(2:end,:); U(end,:)]; % bottom (u(i,j)->u(i+1,j),u(N,j) -> u(N,j))

% second component
Vr = [V(:,2:end) V(:,end)]; % right (v(i,j)->v(i,j+1),v(i,N) -> v(i,N))
Vl = [V(:,1) V(:,1:end-1)]; % left (v(i,j)->v(i,j-1),v(i,1) -> v(i,1))
Vt = [V(1,:); V(1:end-1,:)]; % top (v(i,j)->v(i-1,j),v(1,j) -> v(1,j))
Vb = [V(2:end,:); V(end,:)]; % bottom (v(i,j)->v(i+1,j),v(N,j) -> v(N,j))

% putting all together
Uf = nu1/hx^2*(Ur - 2*U +  Ul) + nu1/hy^2*(Ut - 2*U + Ub) + f(U,V,par);
Vf = nu2/hx^2*(Vr - 2*V +  Vl) + nu2/hy^2*(Vt - 2*V + Vb) + g(U,V,par);

% packing the data into result column vector
result = zeros(2*dim,1);
result(1:dim) = reshape(Uf',1,dim);
result(dim+1:2*dim) = reshape(Vf',1,dim);

end

