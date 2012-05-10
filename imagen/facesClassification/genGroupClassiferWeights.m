% Generate group classifier weights
clear all;
% close all;

localPath = '/volatile/bernardng/data/imagen/';
netwPath = '/home/bn228083/';
addpath(genpath([netwPath,'matlabToolboxes/general']));

fid = fopen([localPath,'subjectLists/facesList.txt']);
nSubs = 58;
sublist = cell(nSubs,1);
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end

method = 5;

if method == 1
    load('wLR');
    w = wLR;
elseif method == 2
    load('wSVM');
    w = wSVM;
elseif method == 3
    load('wRidge');
    w = wRidge;
elseif method == 4
    load('wLASSO');
    w = wLASSO;
elseif method == 5
    load('wEN');
    w = wEN;
elseif method == 6
    load('wAnat');
    w = wAnat;
elseif method == 7
    load('wAnatGroup');
    w = wAnatGroup;
end

[nFeat,nFolds,nSubs] = size(w);
nROIs = nFeat-1; % Skip the intercept;
wAve = zeros(1,nROIs,nSubs);
% Normalization
for sub = 1:nSubs
    w(1:nROIs,:,sub) = w(1:nROIs,:,sub)./(ones(nROIs,1)*std(w(1:nROIs,:,sub)));
    wAve(1,:,sub) = squeeze(mean(w(1:nROIs,:,sub),2));
end
disp('Done Normalization');
% sig = maxtpermTestGroup(wAve,{1,[]},0:0.25:100);

    