addpath ../

%% Initialization

disp('ROCK2 numerical solver for the CDIMA model');

disp('Beginning the compilation...');

%% Compilation

mexcuda NVCCFLAGS='-arch=sm_52' -R2018a -I../include/ mexfile.cu;

mex_file_handler();

disp('Compilation successful!');

