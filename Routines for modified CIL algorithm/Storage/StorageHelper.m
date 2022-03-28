classdef StorageHelper
    %StorageHelper This class provides methods for easy path
    %management for both local and cloud storage directories
    %   The general idea of local / cloud storage could be formulated as
    %   follows. The data, used by the MATLAB scripts in this library is
    %   saved in prescribed folder (called local storages) and organized
    %   with respect to script and run names. Cloud storage is a subfolder
    %   in Google Drive where lightweight files are copied for easier
    %   remote access.
    
    properties
        LocalStoragePath = '/home/alexey_kazarnikov/MATLAB/Storage/';
        CloudStoragePath = '/home/alexey_kazarnikov/MATLAB/Storage/cloud';
        ScriptName = '';
        FolderName = '';
		PathSeparator = '';
    end
    
    methods
        function obj = StorageHelper(scriptName, folderName)
            % determining correct path separator
            if isunix
				obj.PathSeparator = '/';
			elseif ispc
				obj.PathSeparator = '\';
			else
				error('Platform not supported! Please check and adjust the current file!');
            end
            
            % determining the location of storage folder
            currentFileName = mfilename();
            currentFilePath = mfilename('fullpath');
            currentFilePath = currentFilePath(1:end-length(currentFileName));
            
            storagePath = strcat(currentFilePath,...
                '..',...
                obj.PathSeparator,...
                '..',...
                obj.PathSeparator,...
                'Storage');
            
            obj.LocalStoragePath = storagePath;
            obj.CloudStoragePath = strcat(storagePath,...
                obj.PathSeparator,...
                'cloud');
            
            obj.ScriptName = scriptName;
            obj.FolderName = folderName;
			
        end
        
        function resultPath = createPath(obj,fileName,storagePath)
            scriptPath = strcat(storagePath, obj.PathSeparator, obj.ScriptName);
            folderPath = strcat(scriptPath, obj.PathSeparator, obj.FolderName);
            if ~exist(folderPath,'dir')
                mkdir(scriptPath, obj.FolderName);
            end
            
            resultPath = strcat(folderPath, obj.PathSeparator, fileName);
        end
        
        function resultPath = createLocalPath(obj,fileName)
            resultPath = obj.createPath(fileName,obj.LocalStoragePath);
        end
        
        function resultPath = createCloudPath(obj,fileName)
            resultPath = obj.createPath(fileName,obj.CloudStoragePath);
        end
    end
end

