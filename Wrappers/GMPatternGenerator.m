function S = GMPatternGenerator(T0, T1, dT, InitialStep, par, IC, Nsim, devices)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

N = sqrt(size(IC,1) / 2);

if N ~= 32 && N ~= 64
    error('The resolution of spatial grid has to be equal to 32 or 64!');
end

cudaMexExists = false;
if N == 32 && exist('GMPatternGeneratorX32')
    cudaMexExists = true;
end
if N == 64 && exist('GMPatternGeneratorX64')
    cudaMexExists = true;
end

if cudaMexExists
    if N == 32
        S = GMPatternGeneratorX32(...
            T0,...
            T1,...
            dT,...
            InitialStep,...
            par,... 
            IC,... 
            Nsim,...
            devices);
    else
        S = GMPatternGeneratorX64(...
            T0,...
            T1,...
            dT,...
            InitialStep,...
            par,... 
            IC,... 
            Nsim,...
            devices);
    end
elseif exist('GMPatternGeneratorCPP')
    S = GMPatternGeneratorCPP(...
        T0,...
        T1,...
        N,...
        InitialStep,...
        par,...
        IC,...
        Nsim);
else
    f = @(U,V,p) U * (p.rhoa * U ./ V - p.mua);
    g = @(U,V,p) p.rhoi * U.*U - p.mui * V;
    S = RDPatternGenerator(f,g,T0,T1,par,IC,Nsim,N);
end

end
