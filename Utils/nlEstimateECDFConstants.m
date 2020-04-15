function [R0,base] = nlEstimateECDFConstants(dist,nr,uniform)

if ~uniform
    Rmax = max(max(dist));
    Rmin = min(min(dist));
    R0 = Rmax;
    base = (Rmax/Rmin).^(1/nr);
else
    Rmax = max(max(dist));
    R0 = Rmax;
    base = R0 / (nr + 1);
end

end

