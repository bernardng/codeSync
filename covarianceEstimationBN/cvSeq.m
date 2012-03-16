% Generates cross validation fold in sequential blocks
% Example: (123)(456)(789)(10)
function [trainInd,testInd] = cvSeq(nSamples,kFolds)
fold = cell(kFolds,1);
foldLen = round(nSamples/kFolds); % Length of each fold
remain = nSamples-(kFolds-1)*foldLen; % if nSamples/kFolds != int
% Divide 1:nSamples into kFolds segments
for i = 1:kFolds
    if i < kFolds
        fold{i} = (i-1)*foldLen+1:i*foldLen;
    else
        fold{i} = (i-1)*foldLen+1:(i-1)*foldLen+remain;
    end
end
% Concatenate segments together to generate training and test indices
trainInd = cell(kFolds,1);
testInd = cell(kFolds,1);
for i = 1:kFolds
    ind = find(1:kFolds~=i);
    for j = 1:length(ind)
        trainInd{i} = [trainInd{i},fold{ind(j)}];
    end
    testInd{i} = fold{i}; 
end