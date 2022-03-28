% this helper method is used to extract arguments from MATLAB varargin structure 
%   INPUT
%   vararg : varargin cell array
%   k : index of the array to extract
%   OUTPUT
%   val : extracted value
function val = nlExtractArg(vararg,k)
    if length(vararg) >= k
        val = vararg{k};
    else
        val = [];
    end
end

