function [g] = nlCreateDEGraphics(it,x,itmax,names,indx)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

D = size(x,2);
for k=1:D
    subplot(D,1,k)
    hold on
    if (nargin > 4)
        g{k} = plot(it,x(:,k),'b.',it,x(indx,k),'r.');
    else
        g{k} = plot(it,x(:,k),'b.');
    end
    xlim([0 itmax])
    hold off
    title(names{k})
end

end

