function [distance] = DistanceMatrix(S1,S2)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if exist('DistanceMatrixMEX')
    distance = DistanceMatrixMEX(S1,S2);
else
    distance = dist(S1',S2);
end

end

