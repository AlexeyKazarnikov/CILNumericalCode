function Soutput = pdSet(S, grid, Sdata, compIndices, dataIndices)
%pdSet Updates pattern data in a two-dimensional array based on specified 
% component and data indices.
%INPUT
%S : Two-dimensional array containing the pattern data.
%grid : Structure, that contain spatial desription of the data.
%Sdata : Two-dimensional array containing the data to set.
%compIndices : Array of indices specifying which components to set.
%dataIndices : (Optional) Array of indices specifying which data columns 
% to set. Defaults to all columns.
%OUTPUT
%Soutput : Multi-dimensional array with the specified data set at the 
% given indices.


Nelem = grid.numel();
Ncomp = length(compIndices);
Ndata = size(S, 2);

if nargin < 5 || isempty(dataIndices)
    dataIndices = 1 : Ndata;
end

Soutput = S;
for iComp = 1 : Ncomp
    iStart = (iComp - 1) * Nelem + 1;
    iEnd = iComp * Nelem;
    jStart = (compIndices(iComp) - 1) * Nelem + 1;
    jEnd = compIndices(iComp) * Nelem;  

    Soutput(jStart : jEnd, dataIndices) = Sdata(iStart : iEnd, :);
end

end