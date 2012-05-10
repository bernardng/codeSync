% Load features and labels
% Input:    subject = subject#
% Output:   features = NxD matrix, N = #samples, D = #features
%           labels = Nx1 vector, N = #samples            
% Note: Taking 9 out of 10 volumes associated with face images to ensure equal
% #samples with control (triangle) image

function [features,labels] = loadFeatLabel(subject)
localPath = '/volatile/bernardng/data/imagen/';
% Manually determined parameters
thresh = 0.56;
nSampTrial = 8;

load([localPath,subject,'/facesfMRI/tc_task_parcel500.mat']);
tc = tc_parcel; clear tc_parcel;
regressors = load([localPath,subject,'/facesfMRI/facesSPM.mat']);
regressors = regressors.SPM.xX.X;
ind = regressors(:,2) > thresh; % Take more neutral face stimulus
features = tc(ind,:);
labels = ones(nSampTrial,1);
indCtrl = find(regressors(:,11) > thresh);
features = [features;tc(indCtrl(1:nSampTrial),:)];
% ind = regressors(:,6) > thresh;
% features = [features;tc(ind,:)];
labels = [labels;-ones(nSampTrial,1)];
for i = 2:9 % 2 to 9 since only 9 trials for control images
    ind = regressors(:,i+1) > thresh;
    features = [features;tc(ind,:)];
    labels = [labels;ones(nSampTrial,1)];
    features = [features;tc(indCtrl((i-1)*nSampTrial+1:nSampTrial+(i-1)*nSampTrial),:)];
%     ind = regressors(:,i+5) > thresh;
%     features = [features;tc(ind,:)];
    labels = [labels;-ones(nSampTrial,1)];
end
features = [features,ones(size(features,1),1)]; % For learning the intercept