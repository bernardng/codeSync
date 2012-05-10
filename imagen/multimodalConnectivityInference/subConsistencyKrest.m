% Assessment of subject consistency on Krest
% clear all; 
close all;
localpath = '/volatile/bernardng/';
netwpath = '/home/bn228083/';
addpath(genpath([netwpath,'matlabToolboxes/general']));

fid = fopen([localpath,'data/imagen/subjectLists/subjectListDWI.txt']);
nSubs = 60;
sublist = cell(nSubs,1);
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end

method = 2; % 1=GL, 2=WGL, 3=OAS
template = 1; % 1=parcel500, 2=FS ROIs
if method == 1
    methName = 'GL';
elseif method == 2
    methName = 'WGL';
else
    methName = 'OAS';
end
if template == 1
    templateName = '_parcel500';
else
    templateName = '_roi_fs';
end
dice = zeros(nSubs);
klDiv = zeros(nSubs);
affInvDist = zeros(nSubs);
nROIs = 491;
% Indices of upper triangle of a matrix for DICE
% upperTri = find(triu(ones(nROIs),1));
if 0
for subi = 1:nSubs
    if method == 1
        load([localpath,'data/imagen/',sublist{subi},'/restfMRI/K_rest',templateName,'_quic335_cv.mat']);
    elseif method == 2
        load([localpath,'data/imagen/',sublist{subi},'/restfMRI/K_rest_anatDense',templateName,'_quic335_cv.mat']);
    else
        load([localpath,'/data/imagen/',sublist{subi},'/restfMRI/tc_rest',templateName,'.mat']);
        if template == 1
            tcRest = tc_parcel; clear tc_parcel;
        else
            tcRest = tc_roi; clear tc_roi;
        end
        tcRest = tcRest-ones(size(tcRest,1),1)*mean(tcRest);
        tcRest = tcRest./(ones(size(tcRest,1),1)*std(tcRest));
        % Insert random signal for zero time courses
        indNan = isnan(tcRest);
        if sum(indNan(:))~=0
            tcRest(indNan) = randn(sum(indNan(:)),1);
        end
        Krest = inv(oas(tcRest));
    end
    Kresti = Krest;
   
    for subj = 1:nSubs
        if method == 1
            load([localpath,'data/imagen/',sublist{subj},'/restfMRI/K_rest',templateName,'_quic335_cv.mat']);
        elseif method == 2
            load([localpath,'data/imagen/',sublist{subj},'/restfMRI/K_rest_anatDense',templateName,'_quic335_cv.mat']);
        else
            load([localpath,'/data/imagen/',sublist{subj},'/restfMRI/tc_rest',templateName,'.mat']);
            if template == 1
                tcRest = tc_parcel; clear tc_parcel;
            else
                tcRest = tc_roi; clear tc_roi;
            end
            tcRest = tcRest-ones(size(tcRest,1),1)*mean(tcRest);
            tcRest = tcRest./(ones(size(tcRest,1),1)*std(tcRest));
            % Insert random signal for zero time courses
            indNan = isnan(tcRest);
            if sum(indNan(:))~=0
                tcRest(indNan) = randn(sum(indNan(:)),1);
            end
            Krest = inv(oas(tcRest));
        end
        Krestj = Krest;

        dice(subi,subj) = diceCoef(Kresti(:)~=0,Krestj(:)~=0);
%         klDiv(subi,subj) = kullbackLeiblerDivergence(Kresti,Krestj);
%         affInvDist(subi,subj) = affineInvariantMatrixDist(Kresti,Krestj);
    end
    disp(['Done subject',int2str(subi)]);
end
dice = dice(~eye(nSubs));
% klDiv = klDiv(~eye(nSubs));
% affInvDist = affInvDist(~eye(nSubs));

% save(['ConsistencyDice',methName],'dice');
% save(['ConsistencyKLDiv',methName,templateName],'klDiv');
% save(['ConsistencyAffInvDist',methName,templateName],'affInvDist');
end

% Permuting the columns and rows
% Comparisons with random symmetric matrices
nPerm = 200;
Krand = zeros(nROIs,nROIs,nSubs,nPerm);
for sub = 1:nSubs
    load([localpath,'data/imagen/',sublist{sub},'/restfMRI/K_rest_anatDense',templateName,'_quic335_cv.mat']);
    for k = 1:nPerm
        ind = randperm(nROIs);
        Krand(:,:,sub,k) = Krest(ind,ind);
    end
    disp(['Done subject',int2str(sub)]);
end

dice = zeros(nSubs,nSubs,nPerm);
for k = 1:nPerm
    for subi = 1:nSubs
        for subj = 1:nSubs
            supp1 = Krand(:,:,subi,k)~=0;
            supp2 = Krand(:,:,subj,k)~=0;
            dice(subi,subj,k) = diceCoef(supp1(:),supp2(:));
        end
    end
end
diceAve = mean(dice,3);
diceAve = diceAve(~eye(nSubs));

if 0 
% Comparisons with random symmetric matrices
Krand = zeros(nROIs,nROIs,nSubs);
for sub = 1:nSubs
    load([localpath,'data/imagen/',sublist{sub},'/restfMRI/K_rest_anatDense',templateName,'_quic335_cv.mat']);
    prcNonzero = sum(Krest(:)~=0)/length(Krest(:));
    % Testing DICE with random matrix with similar # of nonzero elements
    Krest = randn(nROIs);
    Krest = Krest+Krest';
    [dummy,ind] = sort(abs(Krest(:)));
    Krest(ind(1:round((1-prcNonzero)*length(Krest(:))))) = 0;
    Krand(:,:,sub) = Krest;
    disp(['Done subject',int2str(sub)]);
end

dice = zeros(nSubs);
for subi = 1:nSubs
    for subj = 1:nSubs
        supp1 = Krand(:,:,subi)~=0;
        supp2 = Krand(:,:,subj)~=0;
        dice(subi,subj) = diceCoef(supp1(:),supp2(:));
    end
end
dice = dice(~eye(nSubs));        
end        