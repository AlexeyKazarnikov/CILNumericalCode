function [Sout, gridOut] = pdResize2(S, grid, newSize)
%pdResize2 Resizes two-dimensional pattern data to a new resolution.
%INPUT
%S : Two-dimensional array containing the pattern data.
%grid : Structure, that contain spatial desription of the data.
%newSize : New resolution to which the pattern data should be resized.
%OUTPUT
%Sout : Two-dimensional array containing the resized data.
%gridOut : Structure representing the new grid after resizing.



if grid.dim() ~= 2
    error('Pattern data must be two-dimensional!');
end

if length(newSize) == 1
    newSize = ones(2, 1) * newSize;
end

Nset = size(S, 2);
Ncomp = size(S, 1) / grid.numel();

gridOut = UniformGrid(newSize);
Sout = zeros(Ncomp * gridOut.numel(), Nset);

for iData = 1 : Nset
    for iComp = 1 : Ncomp
        Scomp = pdGet(S, grid, iComp, iData);  
        I = reshape(Scomp, grid.dims()');        
        J = imresize(I, newSize);
        outRange = pdGetRange(gridOut, iComp);
        Sout(outRange, iData) = J(:);
    end
end

end