function g = pdPlot2(S, conf, iData, iDim)
%pdPlot2 Generates a 2D surface plot of the specified pattern data.
%INPUT
%S : Two-dimensional array containing the pattern data.
%conf : Structure, that contain spatial desription of the data.
%iData : (Optional) Index of the data column to plot. Defaults to 1.
%iDim : (Optional) Index of the dimension to plot. Defaults to 1.
%OUTPUT
%g : Handle to the generated surface plot.


if conf.dim() ~= 2
    error('Pattern data must be two-dimensional!')
end

if nargin < 3 || isempty(iData)
    iData = 1;
end

if nargin < 4 || isempty(iDim)
    iDim = 1;
end

Sdata = S(:, iData);
Sdim = pdGet(Sdata, conf, iDim);
Sgrid = reshape(Sdim, conf.dims()');

xrange = linspace(0, conf.ranges(1), conf.dims(1));
yrange = linspace(0, conf.ranges(2), conf.dims(2));

g = surf(xrange, yrange, Sgrid);
set(g, 'LineStyle', 'None');
view([0 90])

end