clc
clear all
close all

%% Initialization

mexName = @(fileName) sprintf('%s.%s',fileName,mexext());

disp('Beginning the compilation...');

%% Distance matrix module

disp('Distance matrix module...');

mexcuda('DistanceMatrixCUDA.cu', '-R2018a', '-lcublas');

mexfile = mexName('../../DistanceMatrixPowL2CUDA');

if isfile(mexfile)
    delete(mexfile);
end

movefile(...
    mexName('DistanceMatrixCUDA'),...
    mexfile...
);

disp('All done!');
