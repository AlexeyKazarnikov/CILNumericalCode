function [mu, C, khi_c, khi_n] = runChi2Test(Y, nbins)
%runChi2Test performs a Chi-squared test on the provided data.
%INPUT
%Y : Multi-dimensional array containing the data.
%nbins : (Optional) Number of bins for the histogram. Defaults to 20.
%OUTPUT
%mu : Mean of the data.
%C : Covariance matrix of the data.
%khi_c : (Optional) Bin centers of the Chi-squared values.
%khi_n : (Optional) Normalized histogram counts of the Chi-squared values.


if nargin < 2
    nbins = 20;
end

mu = mean(Y, 2);
C = cov(Y');

if nargout > 2
    Y0 = Y - repmat(mu, 1, size(Y, 2));
    khi2 = sum(Y0 .* (C \ Y0)); 
    [khi, khi_c] = hist(khi2, nbins);
    khi_n = khi / sum(khi) / (khi_c(2) - khi_c(1));
end

end