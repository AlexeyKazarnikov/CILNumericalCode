function [rngindex1, rngindex2] = nlGetThreshold(Y, eps1, eps2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
N = size(Y,2);

for k=1:N
    if (min(abs(Y(:,k)))>=eps1)
        break;
    end
end

rngindex1 = k;

for k=0:N-1
    if (1-min(abs(Y(:,N-k)))>=eps2)
        break;
    end
end

rngindex2 = N-k;

end

