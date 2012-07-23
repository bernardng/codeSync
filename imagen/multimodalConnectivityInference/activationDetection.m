% Integrating resting state connectivity into task activation detection
clear all; 
% close all;
localpath = '/volatile/bernardng/';
netwpath = '/home/bn228083/';
addpath(genpath([netwpath,'matlabToolboxes/general/']));
addpath(genpath([netwpath,'matlabToolboxes/nifti/']));
addpath(genpath([netwpath,'code/covarianceEstimationBN/']));
addpath(genpath([netwpath,'code/bayesianRegressionBN/']));
addpath(genpath([netwpath,'code/fMRIanalysis']));
fid = fopen([localpath,'data/imagen/subjectLists/subjectListDWI.txt']);
nSubs = 60;
sublist = cell(nSubs,1);
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end
nConds = 10; % 10 experimental conditions

method = 6; 

% Parameters
if method == 1
    methName = 'OLS';
elseif method == 2
    methName = 'Ridge';
elseif method == 3
    methName = 'OAS';
elseif method == 4
    methName = 'SGGM';
elseif method == 5
    methName = 'WSGGM';
elseif method == 6
    methName = 'DWI';
elseif method == 7
    methName = 'sketch';
end
thresh = 0:0.25:100; % For thresholding the t-maps

% Load ROI template
template = load([localpath,'data/imagen/group/fs_parcel500.mat']);
rois = unique(template.template);
nROIs = length(rois)-1; % Skipping background
clear template;

% Activation detection
beta = zeros(nConds,nROIs,nSubs);
for sub = 1:nSubs
    % Load task data
    load([localpath,'data/imagen/',sublist{sub},'/gcafMRI/tc_fs_parcel500.mat']);
    tcTask = tc; clear tc;
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
        load([localpath,'data/imagen/',sublist{sub},'/restfMRI/tc_fs_parcel500.mat']);
        tcRest = tc; clear tc;
        tcRest = tcRest-ones(size(tcRest,1),1)*mean(tcRest);
        tcRest = tcRest./(ones(size(tcRest,1),1)*std(tcRest));
        % Insert random signal for zero time courses
        indNan = isnan(tcRest);
        if sum(indNan(:))~=0
            tcRest(indNan) = randn(sum(indNan(:)),1);
        end
        L = inv(oas(tcRest));
        beta(:,:,sub) = bayesianRegression(X,Y,L)';
    elseif method == 4 % Sparse Gaussian graphical model
        load([localpath,'data/imagen/',sublist{sub},'/restfMRI/K_fs_parcel500_quic335_cv.mat']);
        L = K; clear K;
        beta(:,:,sub) = bayesianRegression(X,Y,L)';
    elseif method == 5 % Weighted sparse Gaussian graphical model
        load([localpath,'data/imagen/',sublist{sub},'/multimodalConn/results_ttk/K_fibcnt_gaussian_blur_fs_parcel500_quic3355_cv.mat']);
        L = K; clear K;
        beta(:,:,sub) = bayesianRegression(X,Y,L)';        
    elseif method == 6 % Anatomical connectivity as prior
        load([localpath,'data/imagen/',sublist{sub},'/dwi/results_ttk/K_gaussian_blur_fs_parcel500.mat']);
        L = diag(sum(Kfibcnt))-Kfibcnt;
        
        load([localpath,'data/imagen/',sublist{sub},'/dwi/results_ukf/K_gaussian_blur_fs_parcel500.mat']);
        L = L + diag(sum(Kfibcnt))-Kfibcnt;
        
        beta(:,:,sub) = bayesianRegression(X,Y,L)';
    elseif method == 7 % Sketch of connectivity pattern as prior
        load([localpath,'data/imagen/',sublist{sub},'/restfMRI/tc_fs_parcel500.mat']);
        tcRest = tc; clear tc;
        tcRest = tcRest-ones(size(tcRest,1),1)*mean(tcRest);
        tcRest = tcRest./(ones(size(tcRest,1),1)*std(tcRest));
        % Insert random signal for zero time courses
        indNan = isnan(tcRest);
        if sum(indNan(:))~=0
            tcRest(indNan) = randn(sum(indNan(:)),1);
        end
        load([localpath,'data/imagen/group/graph_adjacent_fs_parcel500.mat']);
        load([localpath,'data/imagen/group/graph_bilateral_fs_parcel500.mat']);
        A = double(A); B = double(B);
        L = genSynthPrec(tcRest,A,B);
        beta(:,:,sub) = bayesianRegression(X,Y,L)';
    end
    disp(['Subject',int2str(sub),' beta computed']);
end

% Compute t-value of beta contrast
load([localpath,'data/imagen/group/contrastList.mat']);
sig = maxTpermTestGroup(beta,contrastList,thresh,10000);
% Saving contrast of interest results
% save(['sig',methName,'_fs_parcel500.mat'],'sig');



