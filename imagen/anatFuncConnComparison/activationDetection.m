% Integrating resting state connectivity into task activation detection
clear all; 
close all;
addpath(genpath('/home/bernardn/matlabToolboxes/general/'));
addpath(genpath('/home/bernardn/matlabToolboxes/nifti/'));
addpath(genpath('/home/bernardn/code/covarianceEstimationBN/'));
addpath(genpath('/home/bernardn/code/bayesianRegressionBN/'));
addpath(genpath('/home/bernardn/code/fMRIanalysis'));
addpath(genpath('/home/bernardn/code/dMRIanalysis'));
filepath = '/fs/barry/projects/imagen/main/';
fid = fopen([filepath,'subjectLists/subjectListDWI.txt']);
nSubs = 59;
sublist = cell(nSubs,1);
for sub = 1:nSubs
    sublist{sub} = fgetl(fid);
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
    methName = 'SGGMDWI';
end
thresh = 0:0.25:100; % For thresholding the t-maps

% Load ROI template
template = load([filepath,'group/ica_roi_parcel500_refined.mat']);
rois = unique(template.template);
nROIs = length(rois)-1; % Skipping background
clear template;

% Activation detection
beta = zeros(nConds,nROIs,nSubs);
for sub = 1:nSubs
    % Load task data
    load([filepath,sublist{sub},'/gcafMRI/tc_ica_roi_parcel500_refined.mat']);
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
        load([filepath,sublist{sub},'/restfMRI/tc_ica_roi_parcel500_refined.mat']);
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
        load([filepath,sublist{sub},'/restfMRI/K_ica_roi_parcel500_refined_quic335_cv.mat']);
        L = K;
        beta(:,:,sub) = bayesianRegression(X,Y,L)';
    elseif method == 5 % Weighted sparse Gaussian graphical model
        load([filepath,sublist{sub},'/multimodalConn/results_ttk/K_fibcnt_gaussian_blur_ica_roi_parcel500_refined_quic3355_cv.mat']);
        L = K; 
        beta(:,:,sub) = bayesianRegression(X,Y,L)';        
    elseif method == 6 % Anatomical connectivity as prior
        load([filepath,sublist{sub},'/dwi/results_ttk/K_extrap5_ica_roi_parcel500_refined.mat']);
%         load([filepath,sublist{sub},'/dwi/results_ttk/K_gaussian_blur_ica_roi_parcel500_refined.mat']);
        L = diag(sum(Kfibcnt))-Kfibcnt;
        beta(:,:,sub) = bayesianRegression(X,Y,L)';
    elseif method == 7 
        % DWI
        load([filepath,sublist{sub},'/dwi/results_ukf/K_extrap5_ica_roi_parcel500_refined.mat']);
%         load([filepath,sublist{sub},'/dwi/results_ttk/K_gaussian_blur_ica_roi_parcel500_refined.mat']);
        
        % SGGM
        load([filepath,sublist{sub},'/restfMRI/K_ica_roi_parcel500_refined_quic335_cv.mat']);
        
        % OAS
%         load([filepath,sublist{sub},'/restfMRI/tc_ica_roi_parcel500_refined.mat']);
%         tcRest = tc; clear tc;
%         tcRest = tcRest-ones(size(tcRest,1),1)*mean(tcRest);
%         tcRest = tcRest./(ones(size(tcRest,1),1)*std(tcRest));
%         % Insert random signal for zero time courses
%         indNan = isnan(tcRest);
%         if sum(indNan(:))~=0
%             tcRest(indNan) = randn(sum(indNan(:)),1);
%         end
%         K = inv(oas(tcRest));
                
        % AC(FC~=0)
        Kfibcnt(K==0) = 0;
        L = diag(sum(Kfibcnt))-Kfibcnt;
        
        % FC(AC~=0)
%         K(Kfibcnt==0) = 0;
%         [V,D] = eig(K);
%         D = diag(D);
%         K = V*diag(max(D,mean(D>0)))*V';
%         L = K;

        beta(:,:,sub) = bayesianRegression(X,Y,L)';
    end
    disp(['Subject',int2str(sub),' beta computed']);
end

% Compute t-value of beta contrast
load([filepath,'/group/contrastList.mat']);
sig = maxTpermTestOneSampleGroup(beta,contrastList,thresh,10000);
% Saving contrast of interest results
% save(['sig',methName,'_ica_roi_parcel500_refined'],'sig');



