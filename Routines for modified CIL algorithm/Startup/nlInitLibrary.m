function [] = nlInitLibrary()
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

working_dir = pwd();

current_file_name = mfilename();
current_file_path = mfilename('fullpath');
current_file_path = current_file_path(1:end-length(current_file_name));

cd(current_file_path);

addpath ../CostFunctions
addpath ../CIL
addpath ../DistanceProviders
addpath ../../MEX
addpath ../Models
addpath ../Observers
addpath ../Utils
addpath ../Storage

cd (working_dir);

end

