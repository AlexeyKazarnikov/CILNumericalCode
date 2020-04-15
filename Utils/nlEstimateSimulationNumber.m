function [ nsim ] = nlEstimateSimulationNumber( ncurves )
%EstimateSimulationNumber estimate number of simulations, which is needed
%to produce needed amount of pairs
%   INPUT
%   ncurves - desired number of pairs
%   OUTPUT
%   nsim - minimal number of simulations
nsim = 1;
n_pairs = 0;

while (n_pairs < ncurves)
    nsim = nsim+1;
    n_pairs = nsim*(nsim-1)/2;
end

end

