classdef StorageManager
    %StorageHelper This class provides methods for easy path
    %management for both local and cloud storage directories
    %   The general idea of local / cloud storage could be formulated as
    %   follows. The data, used by the MATLAB scripts in this library is
    %   saved in prescribed folder (called local storages) and organized
    %   with respect to script and run names. Cloud storage is a subfolder
    %   in Google Drive where lightweight files are copied for easier
    %   remote access.
    
    properties
        LocalStoragePath = '';
        %RootFolderName = '';
        FolderName = '';
        SubFolderName = '';
		PathSeparator = '';
    end
    
    methods
        function obj = StorageManager(folderName, subFolderName)
            % determining correct path separator
            if isunix
				obj.PathSeparator = '/';
			elseif ispc
				obj.PathSeparator = '\';
			else
				error('Platform is not supported! Please check and adjust the current file!');
            end
            
            % determining the location of storage folder
            currentFileName = mfilename();
            currentFilePath = mfilename('fullpath');
            currentFilePath = currentFilePath(1:end-length(currentFileName));
            
            storagePath = strcat(currentFilePath,...       
                'Data');
            
            obj.LocalStoragePath = storagePath;
            
            %obj.RootFolderName = rootFolderName;
            obj.FolderName = folderName;
            obj.SubFolderName = subFolderName;			
        end
        
        function resultPath = createPath(obj,fileName,storagePath)
            %rootFolderPath = strcat(storagePath, obj.PathSeparator, obj.RootFolderName);
            folderPath = strcat(storagePath, obj.PathSeparator, obj.FolderName);
            subFolderPath = strcat(folderPath, obj.PathSeparator, obj.SubFolderName);
            %if ~exist(rootFolderPath,'dir')
            %    mkdir(storagePath, obj.RootFolderName);
            %end
            if ~exist(folderPath,'dir')
                mkdir(storagePath, obj.FolderName);
            end  
            if ~exist(subFolderPath,'dir')
                mkdir(folderPath, obj.SubFolderName);
            end
            
            resultPath = strcat(subFolderPath, obj.PathSeparator, fileName);
        end
        
        function resultPath = createLocalPath(obj,fileName)
            resultPath = obj.createPath(fileName,obj.LocalStoragePath);
        end
        
    end
end

