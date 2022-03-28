function [rngindex1, rngindex2] = nlGetThresholdNew(Y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
N = size(Y,2);

min_val = min(Y);
max_val = max(Y);

for k=1:N
    if min_val(k) > 0
        rngindex1 = k;
        break;
    end
end

for k=1:N
    if max_val(N-k+1) < 1
        rngindex2 = N-k+1;
        break;
    end
end

end

