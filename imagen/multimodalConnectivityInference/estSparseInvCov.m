% Computing sparse inverse covariance 
clear all; close all;
netwpath = '/home/bn228083/code/';
localpath = '/volatile/bernardng/';
addpath(genpath([netwpath,'covarianceEstimationBN']));
addpath(genpath([netwpath,'matlabToolboxes/quic']));
fid = fopen([localpath,'data/imagen/subjectLists/subjectList.txt']);
nSubs = 65;
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end
subs = 1:nSubs; 

% Parameters
dataType = 1; % 1 = rest, 2 = task
kFolds = 3; % Number of cross-validation folds
nLevels = 3; % Number of refinements on lambda grid
nGridPts = 5; % Number of grid points for lambda, need to be odd number

nROIs = 492; % Number of parcels
Kacc = zeros(nROIs,nROIs,length(subs));
lambda = zeros(length(subs),1);
% matlabpool(7);
for sub = subs
% parfor sub = 1:nSubs
    if dataType == 1 % Load rest data
        tc = load([localpath,'data/imagen/',sublist{sub},'/restfMRI/tc_fs_parcel500.mat']);
        Y = tc.tc;
    elseif dataType == 2 % Load task data
        tc = load([localpath,'data/imagen/',sublist{sub},'/gcafMRI/tc_fs_parcel500.mat']);
        Y = tc.tc;
    end
    % Skipping first time point so that timecourses can be evenly divided into 3 folds
    Y(1,:) = [];    
    
    % Normalizing the time courses
    Y = Y-ones(size(Y,1),1)*mean(Y);
    Y = Y./(ones(size(Y,1),1)*std(Y));
    
    % Insert random signal for zero time courses
    indNan = isnan(Y);
    if sum(indNan(:))~=0
        Y(indNan) = randn(sum(indNan(:)),1);
    end
    
    % Sparse Inverse Covariance Estimation
    [Kacc(:,:,sub),lambda(sub)] = sggmCV(Y,kFolds,nLevels,nGridPts);
    disp(['Done subject',int2str(sub)]);
end
% matlabpool('close');

for sub = 1:nSubs
    K = Kacc(:,:,sub);
    lambdaBest = lambda(sub);
    save([localpath,'data/imagen/',sublist{sub},'/restfMRI/K_fs_parcel500_quic',int2str(kFolds),int2str(nLevels),int2str(nGridPts),'_cv.mat'],'K','lambdaBest');
end
