% Load Regressors
% Input:    sub = subjectID from subjectList.txt
% Output:   regressors = nxm task and motion regressors, n = #samples, m = #regressors
function regressors = loadRegressorsTask(sub)
% Load task and motion regressors
localpath = '/volatile/bernardng/';
load([localpath,'data/imagen/',sub,'/gcaSPM.mat']);
% load(strcat(filepath,sub,'/facesSPM.mat')); % Face data regressors
regressors = SPM.xX.X;
% Add cosine to account for temporal drifts
nTpts = size(regressors,1); tr = 2;
t = (1:tr:tr*nTpts)';
T = tr*nTpts;
freqCut = 1/128; % SPM default is 128s
Tcut = 1/freqCut;
for i = 0:floor(2*T/Tcut) % 0 => remove DC
    regressors = [regressors,cos(i*pi*t/T)];
end
