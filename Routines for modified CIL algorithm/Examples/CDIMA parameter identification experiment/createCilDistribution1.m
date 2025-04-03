function djoint = createCilDistribution1(Djoint, cil)
%createCilDistribution1 estimates the distribution of CIL vectors using
%bootstrapping
%   INPUT
%   Djoint: distance matrix, computed for the set of synthetic patterns
%   cil: structure, that contains the settings of the CIL algorithm
%   OUTPUT
%   djoint: cell array, containing CIL distributions for all scalar
%   mappings used

djoint = {};

for iDist = 1 : size(Djoint, 2)
    D = Djoint(:, iDist);
    D = D(:);
     
    curve = estimateEcdfBins( ...
        D, ...
        cil.Nr ...
        );
    
    cilMapping = @(D) createEcdf(D, curve);
    
    Y = runBootstrap1( ...
        D, ...
        length(D), ...
        cil.Ntr, ...
        cilMapping ...
        );
    
    cilRange = estimateCilRange(Y);
    
    distr.cilData = Y;
    distr.cilMapping = cilMapping;
    distr.cilRange = cilRange;

    djoint{iDist} = distr;
end

end