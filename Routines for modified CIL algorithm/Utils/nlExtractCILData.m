function [cil] = nlExtractCILData(simu_data)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if ~isfield(simu_data,'Chi2') % old data format is used
    cil.ind1 = simu_data.rngindex1;
    cil.ind2 = simu_data.rngindex2;
    cil.Yave = simu_data.Yave;
    cil.iC = simu_data.iC;
    cil.Np = simu_data.npatterns;
else
    cil.ind1 = simu_data.Chi2.rngindex1;
    cil.ind2 = simu_data.Chi2.rngindex2;
    cil.Yave = simu_data.mu0;
    cil.C = simu_data.Sigma0;
    cil.Np = simu_data.CIL.N;
    if isfield(simu_data.CIL,'experimental')
        cil.experimental = simu_data.CIL.experimental;
    end
end

end

