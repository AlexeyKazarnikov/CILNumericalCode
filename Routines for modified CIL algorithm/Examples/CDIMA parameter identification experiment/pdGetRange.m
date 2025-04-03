function iRange = pdGetRange(grid, compIndices)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

Nelem = grid.numel();
Ncomp = length(compIndices);

iRange = [];

for iComp = 1 : Ncomp
    iStart = (compIndices(iComp) - 1) * Nelem + 1;
    iEnd = compIndices(iComp) * Nelem;
    iRange = [iRange; iStart : iEnd];
end

end