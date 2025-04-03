function Soutput = pdGet(S, grid, compIndices, dataIndices)
%pdGet Extracts the pattern data from a two-dimensional array based 
% on component and data indices.
%INPUT
%S : Two-dimensional array containing the pattern data.
%grid : Structure, that contain spatial desription of the data.
%compIndices : Array of indices specifying which components to extract.
%dataIndices : (Optional) Array of indices specifying which data columns 
% to extract. Defaults to all columns.
%OUTPUT
%Soutput : The array of extracted data based on the specified component 
% and data indices.


Nelem = grid.numel();
Ndata = size(S, 2);
Ncomp = length(compIndices);

if nargin < 4 || isempty(dataIndices)
    dataIndices = 1 : Ndata;
end

Nout = length(dataIndices);

Soutput = zeros(Ncomp * Nelem, Nout);
for iComp = 1 : Ncomp
    iStart = (compIndices(iComp) - 1) * Nelem + 1;
    iEnd = compIndices(iComp) * Nelem;
    jStart = (iComp - 1) * Nelem + 1;
    jEnd = iComp * Nelem;

    Soutput(jStart : jEnd, :) = S(iStart : iEnd, dataIndices);
end

end