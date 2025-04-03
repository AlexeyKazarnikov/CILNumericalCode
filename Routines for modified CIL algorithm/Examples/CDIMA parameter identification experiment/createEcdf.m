function c = createEcdf(dist, curve)
%createEcdf creates an empirical cumulative distribution function (ECDF) 
% based on given distances and curve structure.
%INPUT
%dist : Array of distances or data points.
%curve : Structure containing the ECDF parameters and mapping function.
%OUTPUT
%c : Matrix representing the ECDF for the input distances.


if isvector(dist)
    dist = dist(:);
end

Ndim = size(dist, 1);
Nset = size(dist, 2);

c = zeros(curve.Nr + 1, Nset);

for r = 0 : curve.Nr 
   m = dist < curve.Rmap(r);
   c(curve.Nr + 1 - r, :) = sum(m, 1) / Ndim;
end


end

