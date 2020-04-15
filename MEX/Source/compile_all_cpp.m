clc
clear all
close all

mexName = @(fileName) sprintf('%s.%s',fileName,mexext());

solverDir = strcat('-I',fullfile(pwd,'CPP'));

disp('Beginning the compilation...');

%% Distance matrix

disp('Distance matrix...');

mex -R2018a  DistanceMatrixMEX.cpp

mexfile = mexName('../DistanceMatrixMEX');

if isfile(mexfile)
    delete(mexfile);
end
movefile(...
    mexName('DistanceMatrixMEX'),...
    mexfile...
);

%% FitzHugh-Nagumo model

disp('FitzHugh-Nagumo model...');

systemDir = strcat('-I',fullfile(pwd,'FHN'));

mex('RDPatternGeneratorCPP.cpp', '-R2018a', systemDir, solverDir);

mexfile = mexName('../FHNPatternGeneratorCPP');

if isfile(mexfile)
    delete(mexfile);
end
movefile(...
    mexName('RDPatternGeneratorCPP'),...
    mexfile...
);

%% Brusselator reaction-diffusion system

disp('Brusselator model...');

systemDir = strcat('-I',fullfile(pwd,'BZ'));

mex('RDPatternGeneratorCPP.cpp', '-R2018a', systemDir, solverDir);

mexfile = mexName('../BZPatternGeneratorCPP');

if isfile(mexfile)
    delete(mexfile);
end
movefile(...
    mexName('RDPatternGeneratorCPP'),...
    mexfile...
);

%% Gierer-Meinhardt model

disp('Gierer-Meinhardt model...');

systemDir = strcat('-I',fullfile(pwd,'GM'));

mex('RDPatternGeneratorCPP.cpp', '-R2018a', systemDir, solverDir);

mexfile = mexName('../GMPatternGeneratorCPP');

if isfile(mexfile)
    delete(mexfile);
end
movefile(...
    mexName('RDPatternGeneratorCPP'),...
    mexfile...
);

disp('All done!')
