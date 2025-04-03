function [Spatch, gridPatch] = pdSlice2(S, grid, window, offset)
%pdSlice2 extracts a 2D slice from two-dimensional pattern data based on 
% a specified window and offset.
%INPUT
%S : Two-dimensional array containing the pattern data.
%grid : Structure, that contain spatial desription of the data.
%window : Size of the window to extract. Can be a scalar or a two-element 
% vector.
%offset : (Optional) Offset for the window extraction. Can be a scalar or 
% a two-element vector. Defaults to [0, 0].
%OUTPUT
%Spatch : Multi-dimensional array containing the extracted data slice.
%gridPatch : Structure representing the grid of the extracted data slice.


if grid.dim() ~= 2
    error('Pattern data must be two-dimensional!');
end

if length(window) == 1
    window = [window window];
end

if length(offset) == 1
    offset = [offset offset];
end

if nargin < 4
    offset = [0; 0];
end

Ncomp = size(S, 1) / grid.numel();
Ndata = size(S, 2);

gridPatch = UniformGrid(window);
Spatch = zeros(gridPatch.numel() * Ncomp, Ndata);

for iComp = 1 : Ncomp
    Scomp = pdGet(S, grid, iComp);
    Sgrid = reshape(Scomp, [grid.dims(); Ndata]');
    Swindow = Sgrid( ...
        offset(1) + (1 : window(1)), ...
        offset(2) + (1 : window(2)), ...
        : ...
        );
    Swindow = reshape(Swindow, gridPatch.numel(), Ndata);
    Spatch = pdSet(Spatch, gridPatch, Swindow, iComp);
end

end