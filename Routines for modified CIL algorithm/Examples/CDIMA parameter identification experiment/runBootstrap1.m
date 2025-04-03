function [cilVectors, cilIndices] = runBootstrap1( ...
    distVector, ...
    N, ...
    Ntr, ...
    cilMapping ...
    )
%runBootstrap1 implements the standard bootstrapping procedure for a
%one-dimensional vector of scalar values and applies a mapping to the
%sub-sampled data
%   INPUT
%   distVector: source vector of scalar values
%   N: number of scalars to draw in one sampled vector
%   Ntr: number of vectors to sample
%   cilMapping: reduction mapping, which is applied to sub-sampled vectors
%   OUTPUT
%   cilVectors: resulting 2D set of transformed sub-sampled vectors
%   cilIndices: indices of the source set of scalars, used to create each
%   vector


Ndist = length(distVector);

Yind = randi(Ndist, N, 1);
Y = cilMapping(distVector(Yind));

cilVectors = zeros(length(Y), Ntr);
cilIndices = zeros(length(Yind), Ntr);

cilVectors(:, 1) = Y;
cilIndices(:, 1) = Yind;

for iVector = 2 : Ntr
    Yind = randi(Ndist, N, 1);
    Y = cilMapping(distVector(Yind));

    cilVectors(:, iVector) = Y;
    cilIndices(:, iVector) = Yind;
end

end