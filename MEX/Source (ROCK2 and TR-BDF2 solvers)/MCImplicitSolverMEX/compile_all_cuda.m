addpath ../

%% Initialization

disp('Mechano-chemical model with one diffusing morphogen');

disp('Beginning the compilation...');

%% Compilation

mexcuda -R2018a -I../include/ -lcublas mexfile.cu;

mex_file_handler();

disp('Compilation successful!');
