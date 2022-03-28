addpath ../

%% Initialization

disp('ROCK2 numerical solver for the FitzHugh-Nagumo model');

disp('Beginning the compilation...');

%% Compilation

mexcuda -R2018a -I../include/ mexfile.cu;

mex_file_handler();

disp('Compilation successful!');
