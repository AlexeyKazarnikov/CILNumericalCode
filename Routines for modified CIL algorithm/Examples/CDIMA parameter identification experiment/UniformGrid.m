classdef UniformGrid
    %UniformGrid stores the spatial configuration of pattern data.
    

    properties (SetAccess = private)
        GridDimensions = [];
        SpatialRanges = [];
    end

    methods
        function obj = UniformGrid(nGrid, ranges)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.GridDimensions = nGrid(:);

            if nargin < 2
                obj.SpatialRanges = ones(size(obj.GridDimensions));
            else
                obj.SpatialRanges = ranges(:);
                if length(obj.SpatialRanges) == 1 && length(obj.GridDimensions) ~= 1
                    obj.SpatialRanges = obj.SpatialRanges * ones(size(obj.GridDimensions));
                end
            end
        end

        function result = dims(obj, index)
            if nargin < 2
                result = obj.GridDimensions;
            else
                result = obj.GridDimensions(index);
            end
            result = result(:);
        end

        function result = ranges(obj, index)
            if nargin < 2
                result = obj.SpatialRanges;
            else
                result = obj.SpatialRanges(index);
            end
            result = result(:);
        end

        function result = steps(obj, index)
            if nargin < 2
                result = obj.SpatialRanges ./ obj.GridDimensions;
            else
                result = obj.SpatialRanges(index) ./ obj.GridDimensions(index);
            end
            result = result(:);
        end

        function result = numel(obj)
            result = prod(obj.GridDimensions);
        end

        function result = dim(obj)
            result = length(obj.GridDimensions);
        end
    end
end