% Load features and labels
% Input:    subject = subject#
% Output:   features = NxD matrix, N = #samples, D = #features
%           labels = Nx1 vector, N = #samples            

function [features,labels] = loadFeatLabel(subject)
localPath = '/volatile/bernardng/data/imagen/';
% Manually determined parameters
thresh = 0.56;
nTrials = 5;
nSampTrial = 8;

load([localPath,subject,'/facesfMRI/tc_task_parcel500.mat']);
tc = tc_parcel; clear tc_parcel;
regressors = load([localPath,subject,'/facesfMRI/facesSPM.mat']);
regressors = regressors.SPM.xX.X;
ind = regressors(:,1) > thresh;
features = tc(ind,:);
labels = ones(nSampTrial,1);
indCtrl = find(regressors(:,11) > thresh);
features = [features;tc(indCtrl(1:nSampTrial),:)];
labels = [labels;-ones(nSampTrial,1)];
for i = 2:nTrials
    ind = regressors(:,i) > thresh;
    features = [features;tc(ind,:)];
    labels = [labels;ones(nSampTrial,1)];
    features = [features;tc(indCtrl((i-1)*nSampTrial+1:nSampTrial+(i-1)*nSampTrial),:)];
    labels = [labels;-ones(nSampTrial,1)];
end
features = [features,ones(size(features,1),1)];