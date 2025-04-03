function curve = estimateEcdfBins( ...
    dist, ...
    nr, ...
    shift ...
    )
%estimateEcdfBins estimates the empirical cumulative distribution 
% function (ECDF) bins.
%INPUT
%dist : Array of distances or data points.
%nr : Desired number of bins in the eCDF.
%shift : (Optional) Padding of edge bins. Defaults to 1.
%OUTPUT
%curve : Structure containing the following fields:
%        - R0 : Maximum radii value of the eCDF.
%        - b : Differences between eCDF bins.
%        - Rmap : Function handle to map values based on selected bins.
%        - Nr : Number of bins used.


if nargin < 3 || isempty(shift)
    shift = 1;
end

dist = dist(:);

[bins, edges] = histcounts(dist, 'Normalization', 'cdf');
centers = (edges(1 : end-1) + edges(2 : end)) / 2;

[unique_bins, ic] = uniquetol(bins);
selected_bins = interp1( ...
    unique_bins, ...
    centers(ic), ...
    linspace(min(unique_bins), max(unique_bins), nr + 1 + 2 * shift) ...
    );

selected_bins = selected_bins((1 + shift) : end - shift);

curve.R0 = max(centers);
curve.b = diff(selected_bins);
curve.Rmap = @(r) selected_bins(end - r);
curve.Nr = nr;

end

