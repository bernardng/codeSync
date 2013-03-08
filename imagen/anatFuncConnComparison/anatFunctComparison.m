% Compare anatomical and functional connectivity
clear all; 
close all;
filepath = 'I:/research/data/imagen/';
addpath(genpath('I:/research/toolboxes/general'));
addpath(genpath('I:/research/code/covarianceEstimationBN'));

fid = fopen([filepath,'subjectLists/subjectListDWI.txt']);
nSubs = 59;
sublist = cell(nSubs,1);
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end

level = 1; % 1=Subject level, 2=Group level
methodfMRI = 1; % 1=corr, 2=OAS, 3=SGGM
methodDWI = 1; % 1=TTK, 2=UKF, 3=TrackVis
gaussBlur = 0; % 0=extra5, 1=Gaussian Blur

template = 'ica_roi_parcel500_refined';
nROIs = 492;
lowtri = tril(ones(nROIs),-1)==1;
load([filepath,'group/',template,'_coords_matrix_space']);
x = coords(:,1);
y = coords(:,2);
z = coords(:,3);
temp = sqrt((x*ones(1,nROIs)-ones(nROIs,1)*x').^2+(y*ones(1,nROIs)-ones(nROIs,1)*y').^2+(z*ones(1,nROIs)-ones(nROIs,1)*z').^2);
parcelDist = temp(lowtri)*3; % 3mm space

if level == 1
    Krest = zeros(sum(lowtri(:)),nSubs);
    Kanat = zeros(sum(lowtri(:)),nSubs);
    KfiblenAcc = zeros(sum(lowtri(:)),nSubs);
    rho = zeros(nSubs,1);
    for sub = 1:nSubs
        if methodfMRI == 1 % Pearson's correlation
            load([filepath,sublist{sub},'/restfMRI/tc_',template,'.mat']);
            tcRest = tc; clear tc;
            tcRest = tcRest-ones(size(tcRest,1),1)*mean(tcRest);
            tcRest = tcRest./(ones(size(tcRest,1),1)*std(tcRest));
            % Insert random signal for zero time courses
            indNan = isnan(tcRest);
            if sum(indNan(:))~=0
                tcRest(indNan) = randn(sum(indNan(:)),1);
            end
            Ktemp = cov(tcRest);
        elseif methodfMRI == 2 % Partial correlation with OAS
            load([filepath,sublist{sub},'/restfMRI/tc_',template,'.mat']);
            tcRest = tc; clear tc;
            tcRest = tcRest-ones(size(tcRest,1),1)*mean(tcRest);
            tcRest = tcRest./(ones(size(tcRest,1),1)*std(tcRest));
            % Insert random signal for zero time courses
            indNan = isnan(tcRest);
            if sum(indNan(:))~=0
                tcRest(indNan) = randn(sum(indNan(:)),1);
            end
            Ktemp = inv(oas(tcRest));
        elseif methodfMRI == 3 % Sparse partial correlation with SGGM
            load([filepath,sublist{sub},'/restfMRI/K_',template,'_quic335_cv.mat']);
            Ktemp = K;
        end
        if methodfMRI > 1
            dia = diag(1./sqrt(Ktemp(eye(nROIs)==1)));
            Ktemp = -dia*Ktemp*dia;
%             Ktemp = Ktemp;
        end
        Krest(:,sub) = Ktemp(lowtri);
        
        if methodDWI == 1 % TTK
            if gaussBlur == 0
                load([filepath,sublist{sub},'/dwi/results_ttk/K_extrap5_',template,'.mat']);
            else
                load([filepath,sublist{sub},'/dwi/results_ttk/K_gaussian_blur_',template,'.mat']);
            end
        elseif methodDWI == 2 % UKF
            if gaussBlur == 0
                load([filepath,sublist{sub},'/dwi/results_ukf/K_extrap5_',template,'.mat']);
            else
                load([filepath,sublist{sub},'/dwi/results_ukf/K_gaussian_blur_',template,'.mat']);
            end
        elseif methodDWI == 3 % TrackVis for group fiber
            if gaussBlur == 0
                load([filepath,'group/groupFiber/group_all/results_trackvis/K_extrap5_',template,'.mat']);
            else
                load([filepath,'group/groupFiber/group_all/results_trackvis/K_gaussian_blur_',template,'.mat']);
            end
        end
        Kanat(:,sub) = Kfibcnt(lowtri);
        KfiblenAcc(:,sub) = Kfiblen(lowtri);
        
%         % Account for distance
%         beta = pinv(parcelDist)*Krest(:,sub);
%         Krest(:,sub) = Krest(:,sub)-parcelDist*beta;
%         beta = pinv(parcelDist)*Kanat(:,sub);
%         Kanat(:,sub) = Kanat(:,sub)-parcelDist*beta;
        
        rho(sub) = corr(Krest(:,sub),log(Kanat(:,sub)+1));
%         rho(sub) = corr(Krest(:,sub),Kanat(:,sub));
    end
elseif level == 2
    Krest = zeros(sum(lowtri(:)),nSubs);
    Kanat = zeros(sum(lowtri(:)),nSubs);
    rho = zeros(nSubs,1);
    for sub = 1:nSubs
        if methodfMRI == 1 % Pearson's correlation
            load([filepath,'group/grpConn/CConcatAll_ica_roi_parcel500_refined.mat']);
            Kgrp = Cgrp;
        elseif methodfMRI == 2 % Partial correlation with OAS
            load([filepath,'group/grpConn/KoasConcatAll_ica_roi_parcel500_refined.mat']);
        elseif methodfMRI == 3 % Sparse partial correlation with SGGM
            load([filepath,'group/grpConn/KsggmConcatAll_ica_roi_parcel500_refined.mat']);
        end
        if methodfMRI > 1
            dia = diag(1./sqrt(Kgrp(eye(nROIs)==1)));
            Kgrp = -dia*Kgrp*dia;
        end
        Krest(:,sub) = Kgrp(lowtri);
        
        if methodDWI == 1 % TTK
            if gaussBlur == 0
                load([filepath,sublist{sub},'/dwi/results_ttk/K_extrap5_',template,'.mat']);
            else
                load([filepath,sublist{sub},'/dwi/results_ttk/K_gaussian_blur_',template,'.mat']);
            end
        elseif methodDWI == 2 % UKF
            if gaussBlur == 0
                load([filepath,sublist{sub},'/dwi/results_ukf/K_extrap5_',template,'.mat']);
            else
                load([filepath,sublist{sub},'/dwi/results_ukf/K_gaussian_blur_',template,'.mat']);
            end
        elseif methodDWI == 3 % TrackVis for group fiber
            if gaussBlur == 0
                load([filepath,'group/groupFiber/group_all/results_trackvis/K_extrap5_',template,'.mat']);
            else
                load([filepath,'group/groupFiber/group_all/results_trackvis/K_gaussian_blur_',template,'.mat']);
            end
        end
        Kanat(:,sub) = Kfibcnt(lowtri);
        rho(sub) = corr(Krest(:,sub),log(Kanat(:,sub)+1));
    end
end    

mean(rho)

% % Pearson's correlation vs distance
% figure;
% plot(parcelDist,mean(Krest,2),'.');
% corr(parcelDist,mean(Krest,2))
% 
% % Fiber count vs fiber length
% ind = Kanat(:)~=0;
% figure;
% plot(KfiblenAcc(ind)/2,Kanat(ind),'.');
% 
% FC vs AC
figure;
plot(mean(Kanat,2),mean(Krest,2),'.');
hold on; plot([0 100],[0 0],'k');

corr(mean(Kanat,2),mean(Krest,2))

% hold on; plot(log(Kanat(~eye(nROIs))+1),(Krest(~eye(nROIs))),'.');
