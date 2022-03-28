function [R0,base] = nlEstimateECDFConstants(dist,nr,uniform,delta)

if nargin < 4
    delta = 1e-4;
end

if ~uniform
    Rmax = max(max(dist));
    Rmin = min(min(dist));
    eps = (Rmax - Rmin) * delta;
    R0 = Rmax + eps;
    base = ((Rmax + eps)/((1 - eps) * Rmin)).^(1/nr);
else
    Rmax = max(max(dist));
    Rmin = min(min(dist));
    eps = (Rmax - Rmin) * delta;
    R0 = Rmax + eps;
    base = (Rmax + eps - Rmin) / nr;
end

assert((1-eps) >= 0,'Radius negativity detected!');

end

