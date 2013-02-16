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
kFolds = 3;
nLevels = 3; 
nGridPts = 5; 

% Concatenate time courses and compute subject-specific connectivity matrices
tcAcc = [];
C = zeros(nParcels,nParcels,nSubs);
% ClogEucl = zeros(nParcels,nParcels,nSubs);
Coas = zeros(nParcels,nParcels,nSubs);
CoaslogEucl = zeros(nParcels,nParcels,nSubs);
Koas = zeros(nParcels,nParcels,nSubs);
KoaslogEucl = zeros(nParcels,nParcels,nSubs);
Csggm = zeros(nParcels,nParcels,nSubs);
CsggmlogEucl = zeros(nParcels,nParcels,nSubs);
Ksggm = zeros(nParcels,nParcels,nSubs);
KsggmlogEucl = zeros(nParcels,nParcels,nSubs);
for sub = 1:nSubs
    disp(['Computing connectivity matrices of subject',int2str(sub)]);
    load([filepath,sublist{sub},'/restfMRI/tc_',template,'.mat']);
    tc = tc-ones(size(tc,1),1)*mean(tc);
    tc = tc./(ones(size(tc,1),1)*std(tc));
    C(:,:,sub) = cov(tc);
%     ClogEucl(:,:,sub) = logm(C(:,:,sub));
    Coas(:,:,sub) = oas(tc);
    CoaslogEucl(:,:,sub) = logm(Coas(:,:,sub));
    Koas(:,:,sub) = inv(Coas(:,:,sub));
    KoaslogEucl(:,:,sub) = logm(Koas(:,:,sub));
    load([filepath,sublist{sub},'/restfMRI/K_',template,'_quic',int2str(kFolds),int2str(nLevels),int2str(nGridPts),'_cv.mat']);
    Csggm(:,:,sub) = inv(K);
    CsggmlogEucl(:,:,sub) = logm(Csggm(:,:,sub));
    Ksggm(:,:,sub) = K;
    KsggmlogEucl(:,:,sub) = logm(Ksggm(:,:,sub));
    tcAcc = [tcAcc;tc(2:end,:)];
end

% Euclidean mean over subjects
Cgrp = mean(C,3);
save([filepath,'group/grpConn/CEuclSubMean_',template],'Cgrp');
Cgrp = mean(Coas,3);
save([filepath,'group/grpConn/CoasEuclSubMean_',template],'Cgrp');
Kgrp = mean(Koas,3);
save([filepath,'group/grpConn/KoasEuclSubMean_',template],'Kgrp');
Cgrp = mean(Csggm,3);
save([filepath,'group/grpConn/CsggmEuclSubMean_',template],'Cgrp');
Kgrp = mean(Ksggm,3);
save([filepath,'group/grpConn/KsggmEuclSubMean_',template],'Kgrp');

% Log Euclidean mean over subjects
% Cgrp = expm(mean(ClogEucl,3));
% save([filepath,'group/grpConn/CLogEuclSubMean_',template],'Cgrp');
Cgrp = expm(mean(CoaslogEucl,3));
save([filepath,'group/grpConn/CoasLogEuclSubMean_',template],'Cgrp');
Kgrp = expm(mean(KoaslogEucl,3));
save([filepath,'group/grpConn/KoasLogEuclSubMean_',template],'Kgrp');
Cgrp = expm(mean(CsggmlogEucl,3));
save([filepath,'group/grpConn/CsggmLogEuclSubMean_',template],'Cgrp');
Kgrp = expm(mean(KsggmlogEucl,3));
save([filepath,'group/grpConn/KsggmLogEuclSubMean_',template],'Kgrp');

% Compute concatenated subject mean
tcAcc = tcAcc-ones(size(tcAcc,1),1)*mean(tcAcc);
tcAcc = tcAcc./(ones(size(tcAcc,1),1)*std(tcAcc));
Cgrp = cov(tcAcc);
save([filepath,'group/grpConn/CConcatAll_',template],'Cgrp');
Cgrp = oas(tcAcc);
save([filepath,'group/grpConn/CoasConcatAll_',template],'Cgrp');
Kgrp = inv(oas(tcAcc));
save([filepath,'group/grpConn/KoasConcatAll_',template],'Kgrp');
%Kgrp = sggmCV(tcAcc,kFolds,nLevels,nGridPts);
% save([filepath,'group/grpConn/KsggmConcatAll_',template],'Kgrp');
%Cgrp = inv(Kgrp);
%save([filepath,'group/grpConn/CsggmConcatAll_',template],'Cgrp');

% Concatenate subjects into batches
nBatch = 6;
nSubPerBatch = nSubs/nBatch;
C = zeros(nParcels,nParcels,nBatch);
ClogEucl = zeros(nParcels,nParcels,nBatch);
Coas = zeros(nParcels,nParcels,nBatch);
CoaslogEucl = zeros(nParcels,nParcels,nBatch);
Koas = zeros(nParcels,nParcels,nBatch);
KoaslogEucl = zeros(nParcels,nParcels,nBatch);
Csggm = zeros(nParcels,nParcels,nBatch);
CsggmlogEucl = zeros(nParcels,nParcels,nBatch);
Ksggm = zeros(nParcels,nParcels,nBatch);
KsggmlogEucl = zeros(nParcels,nParcels,nBatch);
for b = 1:nBatch
    disp(['Computing connectivity matrices of batch',int2str(b)]);
    tcAcc = [];
    for sub = 1:nSubPerBatch
        load([filepath,sublist{sub+(b-1)*nSubPerBatch},'/restfMRI/tc_',template,'.mat']);
        tc = tc-ones(size(tc,1),1)*mean(tc);
        tc = tc./(ones(size(tc,1),1)*std(tc));
        tcAcc = [tcAcc;tc(2:end,:)];
    end
    tcAcc = tcAcc-ones(size(tcAcc,1),1)*mean(tcAcc);
    tcAcc = tcAcc./(ones(size(tcAcc,1),1)*std(tcAcc));
    C(:,:,b) = cov(tcAcc);
    ClogEucl(:,:,b) = logm(C(:,:,b));
    Coas(:,:,b) = oas(tcAcc);
    CoaslogEucl(:,:,b) = logm(Coas(:,:,b));
    Koas(:,:,b) = inv(Coas(:,:,b));
    KoaslogEucl(:,:,b) = logm(Koas(:,:,b));
    Ksggm(:,:,b) = sggmCV(tcAcc,kFolds,nLevels,nGridPts);
    KsggmlogEucl(:,:,b) = logm(Ksggm(:,:,b));
    Csggm(:,:,b) = inv(Ksggm(:,:,b));
    CsggmlogEucl(:,:,b) = logm(Csggm(:,:,b));
end

% Euclidean mean over subjects
Cgrp = mean(C,3);
save([filepath,'group/grpConn/CEucl_Concat_6batches_10subs_',template],'Cgrp');
Cgrp = mean(Coas,3);
save([filepath,'group/grpConn/CoasEucl_Concat_6batches_10subs_',template],'Cgrp');
Kgrp = mean(Koas,3);
save([filepath,'group/grpConn/KoasEucl_Concat_6batches_10subs_',template],'Kgrp');
Cgrp = mean(Csggm,3);
save([filepath,'group/grpConn/CsggmEucl_Concat_6batches_10subs_',template],'Cgrp');
Kgrp = mean(Ksggm,3);
save([filepath,'group/grpConn/KsggmEucl_Concat_6batches_10subs_',template],'Kgrp');

% Log Euclidean mean over subjects
Cgrp = expm(mean(ClogEucl,3));
save([filepath,'group/grpConn/CLogEucl_Concat_6batches_10subs_',template],'Cgrp');
Cgrp = expm(mean(CoaslogEucl,3));
save([filepath,'group/grpConn/CoasLogEucl_Concat_6batches_10subs_',template],'Cgrp');
Kgrp = expm(mean(KoaslogEucl,3));
save([filepath,'group/grpConn/KoasLogEucl_Concat_6batches_10subs_',template],'Kgrp');
Cgrp = expm(mean(CsggmlogEucl,3));
save([filepath,'group/grpConn/CsggmLogEucl_Concat_6batches_10subs_',template],'Cgrp');
Kgrp = expm(mean(KsggmlogEucl,3));
save([filepath,'group/grpConn/KsggmLogEucl_Concat_6batches_10subs_',template],'Kgrp');

