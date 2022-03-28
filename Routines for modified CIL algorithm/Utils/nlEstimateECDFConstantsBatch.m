function [R0,base] = nlEstimateECDFConstantsBatch(dist,nr,uniform,delta)

if nargin < 4
    delta = 1e-4;
end

Rmax = min(max(dist,[],2));
Rmin = max(min(dist,[],2));
eps = (Rmax - Rmin) * delta;
R0 = Rmax + eps;

if ~uniform   
    base = ((Rmax + eps)/((1 - eps) * Rmin)).^(1/nr);
else
    base = (Rmax + eps - Rmin) / nr;
end

assert((1-eps) >= 0,'Radius negativity detected!');

end

