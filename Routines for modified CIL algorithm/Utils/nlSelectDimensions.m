function Sd_data = nlSelectDimensions(S_data,dims)

dims = dims(:);
if max(dims) > S_data.vdim
    error( ...
        'One of the requested dimensions is larger than the vector dimension of the pattern data!')
end

% getting number of elements in one dimension
D = size(S_data.data,1) / S_data.vdim;

% allocating memory for the output
Sd = zeros(D*length(dims),size(S_data.data,2));

% copying the data to the output matrix
for k = 1:length(dims)
    % range in input matrix
    ind1 = (dims(k)-1)*D+1;
    ind2 = dims(k)*D;
    Sk = S_data.data(ind1:ind2,:);

    % range in output matrix
    ind1 = (k-1)*D+1;
    ind2 = k*D;
    Sd(ind1:ind2,:) = Sk;
end

if isa(S_data.data,'single')
    Sd = single(Sd);
end

% generating an output pattern data structure
Sd_data = nlCreatePatternData( ...
    Sd, ...
    length(dims), ...
    S_data.sdim, ...
    S_data.spart);

end