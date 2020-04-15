function [curve] = nlExtractCurveData(simu_data)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
curve = simu_data.curve;
curve.isUniformDist = simu_data.useUniformDist;
end

