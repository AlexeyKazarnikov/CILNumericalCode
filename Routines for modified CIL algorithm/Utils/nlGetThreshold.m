function [rngindex1, rngindex2] = nlGetThreshold(Y, eps1, eps2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
N = size(Y,2);

if nargin == 3
    for k=1:N
        if std(Y(:,k)) > eps1
            break;
        end
    end

    rngindex1 = k;

    for k=0:N-1
        if std(Y(:,N-k)) > eps2
            break;
        end
    end

    rngindex2 = N-k;
elseif nargin <= 2
    if nargin == 2
        delta = eps1;
    else
        delta = 0.1;
    end
    
    dev_total = std(Y);
    thr = (max(dev_total) - min(dev_total))*delta;
    
    for k=1:N
        if dev_total(k)>thr
            rngindex1=k;
            break;
        end
    end
    
    for k=0:N-1
        if dev_total(N-k)>thr
            rngindex2 = N-k;
            break;
        end
    end   
%     dev_total = sum(abs(Y - repmat(mean(Y),size(Y,1),1)) > repmat(std(Y),size(Y,1),1));
%     for k=1:N
%         if dev_total(k)>thr_count
%             rngindex1=k;
%             break;
%         end
%     end
%     
%     for k=0:N-1
%         if dev_total(N-k)>thr_count
%             rngindex2 = N-k;
%             break;
%         end
%     end
else
    error('Parameter configuration is not supported!')
end

end

