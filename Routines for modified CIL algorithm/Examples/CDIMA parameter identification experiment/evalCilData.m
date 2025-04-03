function y = evalCilData(D, djoint)
%evalCilData Evaluates the CIL likelihood, using the provided distance matrix and likelihood information.
%INPUT
%D : Multi-dimensional array containing the distances between data and simulated patterns.
%djoint : Cell array of structures, each containing the precomputed CIL likelihoods for different scalar mappings.
%OUTPUT
%y : A joint CIL vector.


Nstages = length(djoint);
Ndims = ndims(D);
otherdims = repmat({':'}, 1, Ndims-1);

y = [];

for iStage = 1 : Nstages  
    Di = D(otherdims{:}, iStage);
    yi = djoint{iStage}.cilMapping(Di);
    yi = yi(djoint{iStage}.cilRange);
    y = [y; yi(:)];
end

end