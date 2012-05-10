% Computing sparse inverse covariance 
clear all; close all;
netwpath = '/home/bn228083/code/';
localpath = '/volatile/bernardng/';
addpath(genpath([netwpath,'covarianceEstimationBN']));
fid = fopen([localpath,'data/imagen/subjectLists/subjectListDWI.txt']);
nSubs = 60;

for i = 1:nSubs
    sublist{i} = fgetl(fid);
end
subs = 1:nSubs; 

% Parameters
nROIs = 491; % Number of parcels
KrestAcc = zeros(nROIs,nROIs,length(subs));
KanatAcc = zeros(nROIs,nROIs,length(subs));

for sub = subs
    tc_parcel = load([localpath,'data/imagen/',sublist{sub},'/restfMRI/tc_rest_parcel500.mat']);
    Y = tc_parcel.tc_parcel;

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
    
    KrestAcc(:,:,sub) = inv(oas(Y));
    load([localpath,'data/imagen/',sublist{1},'/dwi/K_anatGroup_parcel500.mat']);    
    KanatAcc(:,:,sub) = Kanat;
    disp(['Done subject',int2str(sub)]);
end

KrestGrp = mean(KrestAcc,3);
KanatGrp = mean(KanatAcc,3);
