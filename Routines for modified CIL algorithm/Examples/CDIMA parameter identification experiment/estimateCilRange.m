function range = estimateCilRange(Y)
%estimateCilRange estimates the range of indices for data values of eCDF 
% vectors (between 0 and 1) that are separated from the tails.
%INPUT
%Y : Two-dimensional array containing the eCDF vectors as columns.
%OUTPUT
%range : Array of indices where data values are between 0 and 1 (separated 
% from edges). If no such range exists, an empty array is returned.


N = size(Y, 1);

min_val = min(Y, [], 2);
max_val = max(Y, [], 2);

rngindex1 = [];
rngindex2 = [];

for k = 1 : N
    if min_val(k) > 0
        rngindex1 = k;
        break;
    end
end

for k = 1 : N
    if max_val(N - k + 1) < 1
        rngindex2 = N - k + 1;
        break;
    end
end

if isempty(rngindex1) || isempty(rngindex2)
    range = [];
else
    range = rngindex1 : rngindex2;
end

end

