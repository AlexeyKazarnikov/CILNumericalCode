function [model,cil,curve] = nlExtractCILData(simu_data)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

model = simu_data.model;

cil.ind1 = simu_data.rngindex1;
cil.ind2 = simu_data.rngindex2;
cil.Yave = simu_data.Yave;
cil.iC = simu_data.iC;
cil.N = simu_data.N;

curve = simu_data.curve;

end

