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
    R = @(R0,base,r) R0/base.^r;
else  
    R = @(R0,base,r) R0 - r * base;
end

Ndim = numel(dist);

% rr0 = R(R0,base,linspace(nr,0,nr+1));
% rr0 = reshape(rr0,1,1,length(rr0));
% rr0 = repmat(rr0,size(dist,1),size(dist,2),1);
% 
% dist0 = repmat(dist,1,1,nr+1);
% m0 = dist0 < rr0;
% csp = max(1,sum(sum(m0)))./Ndim;
c = zeros(1,nr+1);

for r=0:nr 
   m = dist < feval(R,R0,base,r);
   c(nr+1-r)=sum(sum(m))/Ndim;
end

x = 0:nr;

curve.R0 = R0;
curve.base = base;
curve.nr = nr;
curve.uniform = uniform;

end

