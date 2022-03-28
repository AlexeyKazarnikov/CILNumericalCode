function curve = nlExtractCurveData(simu_data)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
if ~isfield(simu_data,'Chi2') % old data format is used
    curve = simu_data.curve;
    curve.isUniformDist = simu_data.useUniformDist;
else
    curve = simu_data.Curve;
    curve.isUniformDist = simu_data.Settings.UseUniformSpacing;
    curve.nr = simu_data.CIL.M;
    curve.base = curve.b;
end

end

