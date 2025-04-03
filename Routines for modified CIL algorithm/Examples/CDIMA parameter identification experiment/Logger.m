classdef Logger
    %Logger provides functionalities to manage and log information to a 
    % file. It includes methods to set text and matrix data.

    methods (Static, Access=public)
      function out = logFileName(data)
         persistent logFileName;
         if nargin
            logFileName = data;
         end
         out = logFileName;
      end

      function log(strData)         
          fileName = Logger.logFileName;  
          if isempty(fileName)
              warning('Log file name is empty! Logging is not possible.');
              return;
          end

          fid = fopen(fileName, 'a+');
          fprintf(fid, "%s    %s", string(datetime), strData);
          fprintf(fid, '\n');
          fclose(fid);
      end

      function logMatrix(data)
          fileName = Logger.logFileName; 
          if isempty(fileName)
              warning('Log file name is empty! Logging is not possible.');
              return;
          end

          logFolderName = char(fileName);
          logFolderName = logFolderName(1 : end - 4);
          if ~exist(logFolderName, 'dir')
              mkdir(logFolderName);
          end

          dateTimeString = string(datetime);
          dateTimeString = strrep(dateTimeString, ' ', '-');
          dateTimeString = strrep(dateTimeString, ':', '-');
          
          dataFileName = strcat( ...
              logFolderName, ...
              filesep, ...
              'data_', ...
              dateTimeString, ...
              '.mat' ...
              );
          save(dataFileName, 'data');
      end
   end
end