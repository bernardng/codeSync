% Compare anatomical and functional connectivity
clear all; close all;
localpath = '/volatile/bernardng/';
netwpath = '/home/bn228083/';
addpath(genpath([netwpath,'matlabToolboxes/general']));

fid = fopen([localpath,'data/imagen/subjectLists/subjectListDWI.txt']);
nSubs = 60;
sublist = cell(nSubs,1);
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end

nROIs = 491;
figure;
for sub = 1:nSubs
    load([localpath,'data/imagen/',sublist{sub},'/restfMRI/K_rest_parcel500_quic335_cv.mat']);
    load([localpath,'data/imagen/',sublist{sub},'/dwi/K_anatDense_parcel500.mat']);
    load([localpath,'/data/imagen/',sublist{sub},'/restfMRI/tc_rest_parcel500.mat']);
    tcRest = tc_parcel; clear tc_parcel;
    tcRest = tcRest-ones(size(tcRest,1),1)*mean(tcRest);
    tcRest = tcRest./(ones(size(tcRest,1),1)*std(tcRest));
    % Insert random signal for zero time courses
    indNan = isnan(tcRest);
    if sum(indNan(:))~=0
        tcRest(indNan) = randn(sum(indNan(:)),1);
    end
    Crest = cov(tcRest);
    hold on; plot(log(Kanat(~eye(nROIs))+1),(Crest(~eye(nROIs))),'.');
end
