function mex_file_handler()

current_path = split(pwd,filesep);
current_mex_name = current_path{end};

mexfile = strcat('../../',current_mex_name,'.',mexext());

if isfile(mexfile)
    delete(mexfile);
end

movefile(...
    strcat('mexfile','.',mexext()),...
    mexfile...
);

end