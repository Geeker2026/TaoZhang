function [W,B,t1,t2] = tdlpp_graph(trainBlocks,trainLabels,t1,t2)
%TDLPP_GRAPH Within-class sample affinity and class-mean affinity graphs.
%CALCULATE_WEIGHTS_DLPP_V2_FROZEN Build the graphs used by TDLPP.
% Two inputs reproduce the V2 automatic widths. Optional positive t1 and
% t2 values support the graph-kernel sensitivity experiments.

    numTrain = numel(trainBlocks);
    trainLabels = trainLabels(:)';
    if numel(trainLabels) ~= numTrain
        error('TDLPP:LabelCountMismatch', ...
            'The number of labels must match the number of training blocks.');
    end
    classes = unique(trainLabels,'sorted');
    classCount = numel(classes);

    distances = zeros(numTrain,numTrain);
    for i = 1:numTrain
        xi = trainBlocks{i};
        for j = i+1:numTrain
            distances(i,j) = norm(xi-trainBlocks{j},'fro');
            distances(j,i) = distances(i,j);
        end
    end

    classMeans = cell(1,classCount);
    for ci = 1:classCount
        idx = find(trainLabels == classes(ci));
        total = zeros(size(trainBlocks{1}));
        for j = idx
            total = total + trainBlocks{j};
        end
        classMeans{ci} = total / numel(idx);
    end

    betweenDistances = zeros(classCount,classCount);
    for ci = 1:classCount
        for cj = ci+1:classCount
            betweenDistances(ci,cj) = ...
                norm(classMeans{ci}-classMeans{cj},'fro');
            betweenDistances(cj,ci) = betweenDistances(ci,cj);
        end
    end

    if nargin < 3 || isempty(t1)
        t1 = mean(distances(triu(true(numTrain),1)));
    end
    if nargin < 4 || isempty(t2)
        t2 = mean(betweenDistances(triu(true(classCount),1)));
    end
    validateattributes(t1,{'numeric'},{'scalar','real','finite','positive'});
    validateattributes(t2,{'numeric'},{'scalar','real','finite','positive'});
    t1 = max(t1,1e-12);
    t2 = max(t2,1e-12);

    W = zeros(numTrain,numTrain);
    for ci = 1:classCount
        idx = find(trainLabels == classes(ci));
        for i = 1:numel(idx)
            for j = i+1:numel(idx)
                ii = idx(i);
                jj = idx(j);
                W(ii,jj) = exp(-(distances(ii,jj)^2)/t1);
                W(jj,ii) = W(ii,jj);
            end
        end
    end

    B = zeros(classCount,classCount);
    for ci = 1:classCount
        for cj = ci+1:classCount
            B(ci,cj) = exp(-(betweenDistances(ci,cj)^2)/t2);
            B(cj,ci) = B(ci,cj);
        end
    end
end
