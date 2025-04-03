function [res, statistics] = cilCfNew(Djoint, Ddata, cil)
%cilCfNew evaluates the mixed mode CIL cost function for a given set of
%pre-computed distances.
%   INPUT
%   Djoint: distance matrix, computed for the set of synthetic patterns
%   Ddata: distance matrix, computed between experimental data and
%   synthetic patterns
%   OUTPUT
%   res: scalar output of the cost function
%   statistics: technical data, which can be helpful for debugging

trialIndices = randperm(size(Djoint, 1));

res = 1e15;

trial_data = {};

for iTrial = 1 : cil.Ntrial
    Dtrial = Djoint(trialIndices(iTrial), :, :);
    Dtrial = squeeze(Dtrial);
    Dtrial(trialIndices(iTrial), :) = [];

    djoint = createCilDistribution1(Dtrial, cil);

    errorFlag = any(cellfun(@(c) isempty(c.cilRange), djoint));
    if errorFlag
        res = 1e15;
        Y = [];
        khi_c = [];
        khi_n = [];
        y = [];
        return;
    end

    Y_trial = cell2mat( ...
    cellfun(@(c) c.cilData(c.cilRange, :), djoint, 'UniformOutput', false)' ...
    );

    [Y_reduced, ~] = uniquetol(Y_trial', 'ByRows', true);
    [Y_reduced, ic] = uniquetol(Y_reduced', 'ByRows', true);

    [mu_trial, C_trial, khi_c_trial, khi_n_trial] = runChi2Test(Y_reduced);

    y_trial = evalCilData(Ddata, djoint);
    res_trial = (y_trial(ic) - mu_trial)' * (C_trial \ (y_trial(ic) - mu_trial));

    data.res = res_trial;
    data.ires = trialIndices(iTrial);
    data.Y = Y_trial;
    data.khi_c = khi_c_trial;
    data.khi_n = khi_n_trial;
    data.y = y_trial;

    trial_data{iTrial} = data;

    if res_trial < res
        res = res_trial;
        ires = trialIndices(iTrial);
        Y = Y_trial;
        khi_c = khi_c_trial;
        khi_n = khi_n_trial;
        y = y_trial;
    end
end

best.res = res;
best.ires = ires;
best.Y = Y;
best.khi_c = khi_c;
best.khi_n = khi_n;
best.y = y;

statistics.data = trial_data;
statistics.best = best;

end