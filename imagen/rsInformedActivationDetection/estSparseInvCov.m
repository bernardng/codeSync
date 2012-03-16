% Computing sparse inverse covariance 
clear all; close all;
netwpath = '/home/bn228083/code/';
localpath = '/volatile/bernardng/';
addpath(genpath([netwpath,'bayesianRegressionBN']));
addpath(genpath([netwpath,'covarianceEstimationBN']));
fid = fopen([localpath,'data/imagen/subjectLists/subjectList.txt']);
nSubs = 65;
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end

% Parameters
dataType = 1; % 1 = rest, 2 = task
paramSelMethod = 1; % 1 = CV, 2 = Model Evidence
optMethod = 1; % 1 = GL via two-metric projection, 2 = (Friedman,2007)
param = 'linear';
kFolds = 3;
model = 1;
nGridPts = 7; % Number of grid points for lambda, need to be odd number
nLevels = 7; % Number of refinements on lambda grid
nROIs = 850;
K = zeros(nROIs,nROIs,nSubs);
matlabpool(5);
parfor sub = 1:nSubs
    if dataType == 1 % Load rest data
        temp = load([localpath,'data/imagen/',sublist{sub},'/tcRestParcel1000']);
        X = loadRegressorsRest(sublist{sub}); 
        Y = temp.tcRest;
    elseif dataType == 2 % Load task data
        temp = load([localpath,'data/imagen/',sublist{sub},'/tcTaskParcel1000']);
        X = loadRegressorsTask(sublist{sub}); 
        Y = temp.tcTask;
    end
    % Normalizing the time courses
    Y = Y-ones(size(Y,1),1)*mean(Y);
    Y = Y./(ones(size(Y,1),1)*std(Y));
    K(:,:,sub) = sparseGGM(Y,paramSelMethod,optMethod,param,kFolds,nLevels,nGridPts);
%     modelEvid = @(V)modelEvidence(X,Y,V,model,param);
%     K = sparseGGM(Y,paramSelMethod,optMethod,param,kFolds,nLevels,nGridPts,modelEvid);
end
matlabpool close;
% Saving results
for sub = 1:nSubs
    Krest = K(:,:,sub);
    save([localpath,'data/imagen/',sublist{sub},'/KrestParcel1000cvProj'],'Krest');
end