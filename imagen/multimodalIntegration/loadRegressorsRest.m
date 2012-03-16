% Load Regressors
% Input:    sub = subjectID from subjectList.txt
% Output:   regressors = nxm motion regressors, n = #samples, m = #regressors
function regressors = loadRegressorsRest(sub)
localpath = '/volatile/bernardng';
regressors = load([localpath,'/data/imagen/',sub,'/restfMRI/restSPM.txt']);
regressors = [regressors,ones(size(regressors,1),1)];
