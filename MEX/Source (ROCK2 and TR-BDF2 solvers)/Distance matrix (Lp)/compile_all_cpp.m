clc
%clear all
%close all

mexName = @(fileName) sprintf('%s.%s',fileName,mexext());

solverDir = strcat('-I',fullfile(pwd,'CPP'));

disp('Beginning the compilation...');

%% Distance matrix (Lp)

disp('Distance matrix (Lp)...');

mex -R2018a CXXFLAGS='$CXXFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp'  DistanceMatrixLpMEX.cpp

mexfile = mexName('../../DistanceMatrixLpMEX');

if isfile(mexfile)
    delete(mexfile);
end
movefile(...
    mexName('DistanceMatrixLpMEX'),...
    mexfile...
);

%% Distance matrix (Lp power)

disp('Distance matrix (Lp power)...');

mex -R2018a CXXFLAGS='$CXXFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp'  DistanceMatrixPowLpMEX.cpp

mexfile = mexName('../../DistanceMatrixPowLpMEX');

if isfile(mexfile)
    delete(mexfile);
end
movefile(...
    mexName('DistanceMatrixPowLpMEX'),...
    mexfile...
);


disp('All done!')
