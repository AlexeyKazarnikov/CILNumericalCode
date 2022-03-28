function [dS] = nlDiffPatterns(S,model)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

if isa(S,'struct')
    m_dim = S.vdim;
    m_sz = S.sdim;
    m_dx = S.spart;
    m_len = size(S.data,2);
else
    m_dim = model.dim(); % determining the number of components
    m_sz = model.size(); % determining the number of spatial dimention in 
    % EACH component
    m_dx = model.dx(); % determining model spatial steps
    m_len = size(S,2);
end
m_sz_dim = [m_sz m_len]; % n-D size of data dimension
m_dim_sz = prod(m_sz_dim(1:end-1));
m_dnum = length(m_sz); % number of partial derivatives (equal to the number 
% of spatial dimensions)


dS = cell(1,m_dnum); % creating cell array for storing the derivatives
for k=1:m_dnum % filling the respective arrays with zeros
    if isa(S,'struct')
        dS{k} = zeros(size(S.data));
    else
        dS{k} = zeros(size(S));
    end
end

for i=1:m_dim % iterating on data dimensions
    if isa(S,'struct')
        Sdim = nlSelectDimensions(S,i);
        Sdim = Sdim.data;
    else
        Sdim = model.select_dimensions(S,i);
    end
    i0 = (i-1)*size(Sdim,1) + 1;
    i1 = i*size(Sdim,1);
    Sdim = reshape(Sdim,m_sz_dim);
    for j=1:m_dnum % iterating on SPATIAL dimensions
        idx = cell(1,length(m_sz_dim)); % creating index cell array for 
        % extending the respective dimension
        for k = 1:length(idx) % filling index array
            idx{k} = 1:m_sz_dim(k);
        end
        idx{j} = m_sz(j); % leaving only last component in the respective 
        % dimension
        Sdim_end = Sdim(idx{:}); % selecting a slice
        dSdim = cat(j,Sdim,Sdim_end); % copying array for differentiating
        dSdim = diff(dSdim,1,j) / m_dx(j);
        dS{j}(i0:i1,:) = reshape(dSdim,m_dim_sz,m_sz_dim(end));
    end
end

if isa(S,'single')
    for j=1:m_dnum
        dS{j} = single(dS{j});
    end
end

if isa(S,'struct')
    for j=1:m_dnum
        dS{j} = nlCreatePatternData( ...
            dS{j}, ...
            S.vdim, ...
            S.sdim, ...
            S.spart);
    end
end

end

