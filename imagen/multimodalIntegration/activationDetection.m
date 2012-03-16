% Integrating resting state connectivity into task activation detection
clear all; 
% close all;
localpath = '/volatile/bernardng/';
netwpath = '/home/bn228083/';
addpath(genpath([netwpath,'matlabToolboxes/general/']));
addpath(genpath([netwpath,'matlabToolboxes/nifti/']));
addpath(genpath([netwpath,'code/covarianceEstimationBN/']));
addpath(genpath([netwpath,'toolboxes/markSchmidtCode/']));
addpath([netwpath,'code/bayesianRegressionBN/']);
fid = fopen([localpath,'data/imagen/subjectLists/subjectListDWI.txt']);
nSubs = 60;
sublist = cell(nSubs,1);
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end
nConds = 10; % 10 experimental conditions

method = 5; 

% Parameters
if method == 1
    methName = 'OLS';
elseif method == 2
    methName = 'Ridge';
elseif method == 3
    methName = 'OAS';
elseif method == 4
    methName = 'GL';
elseif method == 5
    methName = 'WGL';
end
thresh = 0:0.25:100; % For thresholding the t-maps

% Load ROI template
% nii = load_nii([localpath,'/templates/freesurfer/cort_subcort_333.nii']);
% template = nii.img; clear nii;
template = load([localpath,'data/imagen/group/parcel500refined.mat']);
rois = unique(template.template);
nROIs = length(rois)-1; % Skipping background
clear template;

% Activation detection
beta = zeros(nConds,nROIs,nSubs);
for sub = 1:nSubs
    % Load task data
    load([localpath,'data/imagen/',sublist{sub},'/gcafMRI/tc_task_parcel500.mat']);
    tcTask = tc_parcel; clear tc_parcel;
    % Normalizing the time courses
    tcTask = tcTask-ones(size(tcTask,1),1)*mean(tcTask);
    tcTask = tcTask./(ones(size(tcTask,1),1)*std(tcTask));
    % Insert random signal for zero time courses
    indNan = isnan(tcTask);
    if sum(indNan(:))~=0
        tcTask(indNan) = zeros(sum(indNan(:)),1);
    end
    % Load regressors for task data
    regressorsTask = loadRegressorsTask(sublist{sub}); 
    regressorsTask = regressorsTask(:,1:nConds); % Only the first 10 correspond to task
    X = regressorsTask; Y = tcTask; 
    X = X-ones(size(X,1),1)*mean(X);
    X = X./(ones(size(X,1),1)*std(X));
    % beta estimation (nConds x nROIs x nSubs)
    if method == 1 % OLS 
        beta(:,:,sub) = pinv(X)*Y; 
    elseif method == 2 % Ridge
        L = eye(nROIs); % Using identity to test effect of just ridge penalty
        beta(:,:,sub) = bayesianRegression(X,Y,L)';
    elseif method == 3 % OAS
        load([localpath,'/data/imagen/',sublist{sub},'/restfMRI/tc_rest_parcel500.mat']);
        tcRest = tc_parcel; clear tc_parcel;
        tcRest = tcRest-ones(size(tcRest,1),1)*mean(tcRest);
        tcRest = tcRest./(ones(size(tcRest,1),1)*std(tcRest));
        % Insert random signal for zero time courses
        indNan = isnan(tcRest);
        if sum(indNan(:))~=0
            tcRest(indNan) = randn(sum(indNan(:)),1);
        end
        L = inv(oas(tcRest));
        beta(:,:,sub) = bayesianRegression(X,Y,L)';
    elseif method == 4 % Graphical LASSO
        load([localpath,'data/imagen/',sublist{sub},'/restfMRI/K_rest_parcel500_quic335_cv.mat']);
        L = Krest; clear Krest;
        beta(:,:,sub) = bayesianRegression(X,Y,L)';
    elseif method == 5 % Weighted Graphical LASSO
        load([localpath,'data/imagen/',sublist{sub},'/restfMRI/K_rest_anatDenseBin_parcel500_quic335_cv.mat']);
        L = Krest; clear K_rest;
        beta(:,:,sub) = bayesianRegression(X,Y,L)';        
    end
    disp(['Subject',int2str(sub),' beta computed']);
end

% Compute t-value of beta contrast
load([localpath,'data/imagen/contrastList.mat']);
% sig = maxTpermTestGroup(beta,contrastList(9:12,:),thresh);
sig = maxTpermTestGroup(beta,contrastList,thresh,20000);
% Saving contrast of interest results
% save(['sig',methName],'sig');



