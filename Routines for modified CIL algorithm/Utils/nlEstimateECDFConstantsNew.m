function [R0,base] = nlEstimateECDFConstantsNew(dist,nr,uniform,delta,N,Nset,Nrep)

Rmax = 1e15;
Rmin = 0;

for k=1:Nrep
    Nind = randi(length(dist),1,Nset*N);
    Dind = dist(Nind);
    Rmax = min(Rmax,max(Dind));
    Rmin = max(Rmin,min(Dind));
end

if ~uniform
    eps = (Rmax - Rmin) * delta;
    R0 = Rmax + eps;
    base = ((Rmax + eps)/((1 - eps) * Rmin)).^(1/nr);
else
    eps = (Rmax - Rmin) * delta;
    R0 = Rmax + eps;
    base = (Rmax + eps - Rmin) / nr;
end

assert((1-eps) >= 0,'Radius negativity detected!');

end

