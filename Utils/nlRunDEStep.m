function [x,fx,xbest,fxbest] = nlRunDEStep(x,xbest,fx,F,CR,objFun)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

N = size(x,1);
D = size(x,2);

% generating random pairs of population elements
perMatrix = zeros(N,2);
for k = 1:N
    indexes = randperm(N);
    perMatrix(k,:) = indexes(1:2);
end

% generating donors by mutation
v=repmat(xbest,N,1)+F*(x(perMatrix(1:N,1),:)-x(perMatrix(1:N,2),:));

% performing recombination
pr1 = (rand(N,D)<CR);
pr2 = repmat(1:D,N,1) == randi(2,N,1);

vIdx = (pr1 + pr2) ~= 0;
xIdx = 1-vIdx;
u=x.*xIdx+v.*vIdx;

% evaluating the cost function on donor elements
fu = zeros(N,1);
for k=1:N
    fu(k) = objFun(u(k,:));
end

% determining the elements with lower values of the cost function
idx=fu<fx;

% updating population elements
x(idx,1:D)=u(idx,1:D);    

% updating cost function values for all population elements (needed 
% due to the stochasticity of the cost function)
fx = zeros(N,1);
for k=1:N
    fx(k) = objFun(x(k,:));
end

% finding the best candidate
[fxbest,ixbest]=min(fx);
xbest=x(ixbest,1:D);
end

