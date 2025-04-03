function [result, flag, data, Slow] = valObjFun( ...
    theta, ...
    generatorModel, ...
    distanceModel, ...
    Sdata, ...
    grid, ...
    Ndim, ...
    cil, ...
    vec2par ...
    )
%valObjFun evaluates the mixed mode CIL cost function, using various
%validation (early rejection) techniques to improve the performance.
%INPUT
%theta : Parameter vector at which cost function will be evaluated.
%generatorModel : Model used to generate pattern data.
%distanceModel : Model used to calculate distances between patterns.
%Sdata : Two-dimensional array containing the observed (experimental) data.
%grid : Structure, that contain spatial desription of the data.
%Ndim : Resolution of batch in the middle of simulated patterns that will 
% be used in the computations.
%cil : Structure containing the settings of the CIL approach.
%vec2par : Function handle to convert parameter vector to structure with 
% model parameters.
%OUTPUT
%result : Output value of the objective function.
%flag : Indicator flag for different conditions (1: Turing condition 
% failed, 2: Simulation convergence check failed, 3: Data homogeneity 
% check failed).
%data : Data used in the computational process. Can be used for debugging
%purposes.
%Slow : Sliced and transformed simulated patterndata used in the 
% computations.


Logger.log('CALL OBJECTIVE FUNCTION');

flag = 0;

if ~detectTuringSpace(vec2par(theta), 100, 100)
    result = 1e15;
    flag = 1;

    Logger.log('Turing condition failed!');

    Logger.logMatrix({theta});
    Logger.log('Binary data saved.');

    Logger.log('EXIT OBJECTIVE FUNCTION');

    return;
end

Nval = max(1, round(cil.N * 0.1));
Ntrain = cil.N - Nval;

% computing validation set
[Sval, timesVal, rhsVal] = generatorModel.simulate(Nval, [], vec2par(theta));
Logger.log('VALIDATION SET COMPUTED!')
Logger.log(sprintf('MEAN RHS NORM: %e', mean(rhsVal)));
Logger.log( ...
    sprintf( ...
    'TIMES: [%e, %e], mean: %e', ...
    min(timesVal), max(timesVal), mean(timesVal)) ...
    );

% convergence analysis
Ntries = 10;
Nrun = 0;

for iTry = 1 : Ntries
    if mean(rhsVal) <= generatorModel.Settings.conv_norm
        break;
    end

    [Sval, timesVal, rhsVal] = generatorModel.simulate(Nval, Sval, vec2par(theta));
    Logger.log('WARNING: adjusting integration time...');
    Logger.log(sprintf('VALIDATION SET RECOMPUTED (%i / %i)!', iTry, Ntries));
    Logger.log(sprintf('MEAN RHS NORM: %e', mean(rhsVal)));
    Logger.log( ...
        sprintf( ...
        'TIMES: [%e, %e], mean: %e', ...
        min(timesVal), max(timesVal), mean(timesVal)) ...
        );
    Nrun = Nrun + 1;
end

if mean(rhsVal) > 5 * generatorModel.Settings.conv_norm
    Logger.log('Convergence check failed!');

    Slog = Sval(:, 1:3);
    Slog = pdSlice2(Slog, grid, [Ndim Ndim], [Ndim Ndim]);
    Slog = round(255 * Slog);
    Slog = uint8(Slog);
    
    Logger.logMatrix({single(Slog), theta});
    Logger.log('Binary data saved.');

    Logger.log('EXIT OBJECTIVE FUNCTION');

    result = 1e15;
    flag = 2;

    return;
end

try
    if max(rhsVal) > generatorModel.Settings.conv_norm
        nOut = sum(rhsVal > generatorModel.Settings.conv_norm);
        Logger.log( ...
            sprintf( ...
            'WARNING: Convergence was not reached (%i of %i)!', ...
            nOut, ...
            length(rhsVal) ...
            ));
    end
catch ME
    warning('Objective function: logging code threw an exception!');
    disp(ME);
end


delta = mean(max(Sval) - min(Sval));
if delta < 0.25
    result = 1e15;
    flag = 3;

    Logger.log('Homogeneous check failed!');
    
    Slog = Sval(:, 1:3);
    Slog = pdSlice2(Slog, grid, [Ndim Ndim], [Ndim Ndim]);
    Slog = round(255 * Slog);
    Slog = uint8(Slog);
    
    Logger.logMatrix({single(Slog), theta});
    Logger.log('Binary data saved.');

    Logger.log('EXIT OBJECTIVE FUNCTION');

    return;
end

[Strain, timesTrain, rhsTrain] = generatorModel.simulate(Ntrain, [], vec2par(theta));
Logger.log('TRAINING SET COMPUTED!')
Logger.log(sprintf('MAX RHS NORM: %e', max(rhsTrain)));
Logger.log( ...
        sprintf( ...
        'TIMES: [%e, %e], mean: %e', ...
        min(timesTrain), max(timesTrain), mean(timesTrain)) ...
        );

if Nrun > 0
    for iTrain = 1 : Nrun
        [Strain, timesTrain, rhsTrain] = ...
            generatorModel.simulate(Ntrain, Strain, vec2par(theta));
        Logger.log(sprintf('TRAINING SET RECOMPUTED (%i / %i)!', iTrain, Nrun));
        Logger.log(sprintf('MEAN RHS NORM: %e', mean(rhsTrain)));
        Logger.log( ...
            sprintf( ...
            'TIMES: [%e, %e], mean: %e', ...
            min(timesTrain), max(timesTrain), mean(timesTrain)) ...
            );
    end
end



try
    if max(rhsTrain) > generatorModel.Settings.conv_norm
        nOut = sum(rhsTrain > generatorModel.Settings.conv_norm);
        Logger.log( ...
            sprintf( ...
            'WARNING: Convergence was not reached (%i of %i)!', ...
            nOut, ...
            length(rhsTrain) ...
            ));
    end
catch ME
    warning('Objective function: logging code threw an exception!');
    disp(ME);
end

S = [Sval Strain];
observer = MinMaxObserver();
S = observer.transform(S, generatorModel);

% selecting the first component
S = S(1 : size(S, 1) / 2, :);

Nhigh = generatorModel.Settings.N;
Nlow = distanceModel.Settings.N;
Noffset = round((Nhigh - Nlow) / 2);

Slow = pdSlice2(S, grid, [Ndim Ndim], [Noffset Noffset]);

Djoint = distanceModel.distance(Slow, []);
Ddata = distanceModel.distance(Slow, Sdata);

[result, data] = cilCfNew( ...
            Djoint, ...
            Ddata, ...
            cil);

Slog = S(:, round(linspace(1, cil.N, 3)));
Slog = pdSlice2(Slog, grid, [Ndim Ndim], [Noffset Noffset]);
Slog = round(255 * Slog);
Slog = uint8(Slog);

Logger.logMatrix({single(Slog), theta});
Logger.log('Binary data saved.');

Logger.log(sprintf('Cost function output: %f', result));
Logger.log('EXIT OBJECTIVE FUNCTION');

end