function [idx,coeff] = nlCreateBoundaryMask(sz,pow)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if (length(sz) == 1)
    idx = 1:sz(1);
    coeff = ones(1,sz(1));
    coeff(1) = 0.5;
    coeff(end) = 0.5;
elseif (length(sz) == 2)
    row = [ ...
        ones(1,sz(2)) ...
        sz(1)*ones(1,sz(2)) ...
        2:sz(1)-1 ...
        2:sz(1)-1 ...  
        ];

    col = [ ...
        1:sz(2) ...
        1:sz(2) ...
        ones(1,sz(1)-2) ...
        sz(2)*ones(1,sz(1)-2) ...
        ];

    coeff = 0.5 * ones(1,length(row));
    coeff(1) = 0.25;
    coeff(sz(2)) = 0.25;
    coeff(sz(2)+1) = 0.25;
    coeff(2*sz(2)) = 0.25;

    idx = sub2ind(sz,row,col);
else
    error('Dimension not supported!');
end

if pow > 1
    coeff = coeff.^(1/pow);
end

end

