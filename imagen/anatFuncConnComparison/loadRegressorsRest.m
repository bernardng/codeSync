% Load Regressors
% Input:    sub = subjectID from subjectList.txt
% Output:   regressors = nxm motion regressors, n = #samples, m = #regressors
function regressors = loadRegressorsRest(sub)
filepath = '/fs/barry/projects/imagen/main/';
regressors = load([filepath,sub,'/restfMRI/restSPM.txt']);
regressors = [regressors,ones(size(regressors,1),1)];
