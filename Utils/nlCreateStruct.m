function [result] = nlCreateStruct(fields,values)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
result = struct();

if iscell(fields)
    fields = cell2mat(fields);
end

if iscell(values)
    values = cell2mat(values);
end

for k = 1:length(fields)
    result = setfield(result,fields(k),values(k));
end

end

