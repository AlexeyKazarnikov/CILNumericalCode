function [c,x,curve] = nlCreateECDF(dist,nr,uniform,R0,base)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    uniform = false;
end

if (nargin <= 3)
    [R0,base] = estimateECDFConstants(dist,nr,uniform);
end

if ~uniform
    R = @(R0,base,r) R0/base^r;
else  
    R = @(R0,base,r) R0 - r * base;
end

Ndim = numel(dist);
for r=0:nr 
   m = dist < R(R0,base,r);
   cspsum(r+1) = max(1,sum(sum(m)));
   rr(r+1)=r;
   csp(r+1)=cspsum(r+1)/Ndim;
end

x = rr;
c = fliplr(csp);

curve.R0 = R0;
curve.base = base;
curve.nr = nr;
curve.uniform = uniform;

end

