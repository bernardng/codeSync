% Integrating resting state connectivity into task activation detection
clear all; 
% close all;
filepath = '/home/nbernard/Documents/research/';
addpath(genpath(strcat(filepath,'toolboxes/lightspeed')));
addpath(genpath(strcat(filepath,'toolboxes/cvx')));
fid = fopen('subjectList.txt');
nSubs = 65;
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end
% nSubs = 10; % for testing
nConds = 10;

nIter = 1;
for k = 1:nIter % testing if getting detection by chance

% Parameters
lambda = 0.25;
nComps = 30;
nROIs = 145;
% Subcortical regions
sc = 123:136;
% White matter 
wm = [139,143];
% CSF
csf = [137,138,141,145];
% Grey matter
gm = setdiff(1:nROIs,[sc,wm,csf]);
betaOLS = zeros(nConds,length([sc,gm]),nSubs);
betaReg = zeros(nConds,length([sc,gm]),nSubs);
for sub = 1:nSubs
    % Load task data
    load(strcat('/volatile/bernard/imagenData/',sublist{sub},'/tcTaskROI.mat'));
    % Generate regressors for task data
    regressorsTask = genRegressorsTask(sublist{sub}); 
    regressorsTask = regressorsTask(:,1:nConds); % Only the first 10 correspond to task
    % Load rest data
%     load(strcat('/volatile/bernard/imagenData/',sublist{sub},'/KrestROI',int2str(lambda*100)));
    load(strcat('/volatile/bernard/imagenData/',sublist{sub},'/CrestROIreg',int2str(nComps)));
    KrestROI = inv(CrestROIreg);
%     % Random psd matrix to test if results are artificial
%     nElement = sum(KrestROI(:)~=0)/length(KrestROI(:));
%     rc = 1/cond(KrestROI); clear KrestROI;
%     KrestROI = sprandsym(length([sc,gm]),nElement,rc,2);
%     KrestROI = KrestROI/max(abs(KrestROI(:)));
    % Compute OLS beta (nConds x nROIs x nSubs)
    betaOLS(:,:,sub) = geninv(regressorsTask)*tcTaskROI(:,[sc,gm]);    
    % Compute regularized beta (nConds x nROIs x nSubs)
    X = regressorsTask; Y = tcTaskROI(:,[sc,gm]); 
    L = KrestROI; alpha = 1e-2; 
    betaReg(:,:,sub) = betaEstReg(X,Y,L,alpha);
end
clear tcTaskROI KrestROI X Y L;

% Compute t-value of beta contrast
condRange = 1:nConds; 
% (nContrast x nROIs)     
tvalOLS = zeros(nConds*(nConds-1)/2,length([sc,gm]));
tvalReg = zeros(nConds*(nConds-1)/2,length([sc,gm]));
ind = 0; nPerm = 200;
for cond1 = condRange(1):condRange(end-1)
    % (nSubs x nROIs)
    betaOLS1 = squeeze(betaOLS(cond1,:,:))';
    betaReg1 = squeeze(betaReg(cond1,:,:))';
    for cond2 = cond1+1:condRange(end)
        ind = ind+1;
        betaOLS2 = squeeze(betaOLS(cond2,:,:))';
        betaReg2 = squeeze(betaReg(cond2,:,:))';
        % Compute t-value
        % (nConds xROI nROIs)
        [h,p,ci,stat] = ttest(betaOLS1,betaOLS2);
        tvalOLS(ind,:) = stat.tstat;
        [h,p,ci,stat] = ttest(betaReg1,betaReg2);
        tvalReg(ind,:) = stat.tstat;
        % Compute t-value of null distribution
        for perm = 1:nPerm
            betaPermOLS1 = sign(randn(size(betaOLS1))).*betaOLS1;
            betaPermOLS2 = sign(randn(size(betaOLS2))).*betaOLS2;
            betaPermReg1 = sign(randn(size(betaReg1))).*betaReg1;
            betaPermReg2 = sign(randn(size(betaReg2))).*betaReg2;
            [h,p,ci,stat] = ttest(betaPermOLS(:,:,cond1,perm),betaPermOLS(:,:,cond2,perm));
            tvalOLSPerm(:,perm) = stat.tstat;
            [h,p,ci,stat] = ttest(betaPermReg(:,:,cond1,perm),betaPermReg(:,:,cond2,perm));
            tvalRegPerm(:,perm) = stat.tstat;
        end
%         % Display results
%         figure; plot(tvalOLS(ind,:));
%         hold on; plot(tvalReg(ind,:),'r');
        % Compute threshold
        % Region level test
%         for nr = 1:length([sc,gm])
%             threshOLS(ind,nr) = prctile(tvalPermTempOLS(nr,:),99);
%             threshReg(ind,nr) = prctile(tvalPermTempReg(nr,:),99);
%             sigOLS(ind,nr) = tvalOLS(ind,nr)>threshOLS(ind,nr);
%             sigReg(ind,nr) = tvalReg(ind,nr)>threshReg(ind,nr);
%         end
        % Global level test
        maxTvalOLS = max(tvalOLSPerm);
        maxTvalReg = max(tvalRegPerm);
        sigOLS(ind,:) = tvalOLS(ind,:)>prctile(maxTvalOLS,99.99);
        sigReg(ind,:) = tvalReg(ind,:)>prctile(maxTvalReg,99.99);
    end
    clear betaOLS1 betaOLS2 betaReg1 betaReg2;
end
clear betaOLS betaReg betaPermOLS betaPermReg;

% figure; plot(sum(sigOLS,2)); hold on; plot(sum(sigReg,2),'r');

% Diff in # of detections using a sparse random matrix with same degree of
% sparsity as our estimated RS precision matrix and similar cond number
if nIter > 1
    nDetect(k) = sum(sum(sigReg,2)-sum(sigOLS,2))
else
    nDetect = sum(sum(sigReg,2)-sum(sigOLS,2))
end

end