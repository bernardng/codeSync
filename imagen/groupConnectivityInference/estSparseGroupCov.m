% Estimate Sparse Group Covariance using sgggm.m
addpath(genpath('/home/bernardn/code/covarianceEstimationBN'));
filepath = '/fs/apricot2_share2/imagen/';

% Extract subject number
fid = fopen([filepath,'subjectLists/subjectListRest.txt']);
nSubs = 60;
sublist = cell(nSubs,1);
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end
fclose(fid);

% Initialization
template = 'ica_roi_parcel500_refined';
nParcels = 492; % Depends on parcel template used
nTpts = 186; 

% % Concatenate time courses
% tcAcc = zeros(nTpts,nParcels,nSubs);
% for sub = 1:nSubs
%     load([filepath,sublist{sub},'/restfMRI/tc_',template,'.mat']);
%     tc = tc-ones(size(tc,1),1)*mean(tc);
%     tc = tc./(ones(size(tc,1),1)*std(tc));
%     tcAcc(:,:,sub) = tc(2:end,:);
% end

% Concatenate time courses
nBatch = 6;
nSubPerBatch = nSubs/nBatch;
tcAll = [];
for b = 1:nBatch
    tcAcc = [];
    for sub = 1:nSubPerBatch
        load([filepath,sublist{sub+(b-1)*nSubPerBatch},'/restfMRI/tc_',template,'.mat']);
        tc = tc-ones(size(tc,1),1)*mean(tc);
        tc = tc./(ones(size(tc,1),1)*std(tc));
        tcAcc = [tcAcc;tc(2:end,:)];
    end
    tcAll = cat(3,tcAll,tcAcc);
end

% Compute sparse group covariance with 1 left-out subject
nLevels = 3;
kFolds = 3;
nGridPts = 5;
maxIter = 200;
Cgrp = sgggmCV(tcAll,nLevels,kFolds,nGridPts,maxIter);
save(['/fs/apricot2_share2/imagen/group/Csgggm_',int2str(nBatch),'batches_',int2str(nSubPerBatch),'subs_',template],'Cgrp');

exit;