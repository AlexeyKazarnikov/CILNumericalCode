clc
clear all
close all

%% Initialization

mexName = @(fileName) sprintf('%s.%s',fileName,mexext());


solver32Dir = strcat('-I',fullfile(pwd,'DIM32'));
solver64Dir = strcat('-I',fullfile(pwd,'DIM64'));

disp('Beginning the compilation...');

%% FitzHugh-Nagumo model

disp('FitzHugh-Nagumo model...');

systemDir = strcat('-I',fullfile(pwd,'FHN'));

mexcuda('RDPatternGeneratorCUDA.cu', '-R2018a', systemDir, solver32Dir);

mexfile = mexName('../FHNPatternGeneratorX32');

if isfile(mexfile)
    delete(mexfile);
end
movefile(...
    mexName('RDPatternGeneratorCUDA'),...
    mexfile...
);

mexcuda('RDPatternGeneratorCUDA.cu', '-R2018a', systemDir, solver64Dir);

mexfile = mexName('../FHNPatternGeneratorX64');

if isfile(mexfile)
    delete(mexfile);
end
movefile(...
    mexName('RDPatternGeneratorCUDA'),...
    mexfile...
);

%% Brusselator reaction-diffusion system

disp('Brusselator model...');

systemDir = strcat('-I',fullfile(pwd,'BZ'));

mexcuda('RDPatternGeneratorCUDA.cu', '-R2018a', systemDir, solver32Dir);

mexfile = mexName('../BZPatternGeneratorX32');

if isfile(mexfile)
    delete(mexfile);
end
movefile(...
    mexName('RDPatternGeneratorCUDA'),...
    mexfile...
);

mexcuda('RDPatternGeneratorCUDA.cu', '-R2018a', systemDir, solver64Dir);

mexfile = mexName('../BZPatternGeneratorX64');

if isfile(mexfile)
    delete(mexfile);
end
movefile(...
    mexName('RDPatternGeneratorCUDA'),...
    mexfile...
);

%% Gierer-Meinhardt model

disp('Gierer-Meinhardt model...');

systemDir = strcat('-I',fullfile(pwd,'GM'));

mexcuda('RDPatternGeneratorCUDA.cu', '-R2018a', systemDir, solver32Dir);

mexfile = mexName('../GMPatternGeneratorX32');

if isfile(mexfile)
    delete(mexfile);
end
movefile(...
    mexName('RDPatternGeneratorCUDA'),...
    mexfile...
);

mexcuda('RDPatternGeneratorCUDA.cu', '-R2018a', systemDir, solver64Dir);

mexfile = mexName('../GMPatternGeneratorX64');

if isfile(mexfile)
    delete(mexfile);
end
movefile(...
    mexName('RDPatternGeneratorCUDA'),...
    mexfile...
);


disp('All done!');
