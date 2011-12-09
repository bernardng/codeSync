% Simulation of Resting-state/Task data pairs
clear all; close all;
filepath = 'D:/research/';
addpath(genpath([filepath,'toolboxes/l1General']));
addpath(genpath([filepath,'toolboxes/lightspeed']));
addpath(genpath([filepath,'toolboxes/markSchmidtCode']));
addpath(genpath([filepath,'covarianceEstimationNg']));
addpath(genpath([filepath,'bayesianRegressionNg']));
% Parameters
nSubs = 10; 
nROIs = 800; 
Id = eye(nROIs);
nActive = 50;
nNonActive = 50;
nTpts = 200;
dense = 0.05;
gamma = 10; 
sig = 1; 
snr = 0.2; 
noise = sqrt(sig^2/snr);
snrRS = 1;
nIter = 1;
thresh = 0:0.05:1;
nThresh = length(thresh);
% Generate inverse covariance of intrinsic network
t = (1:10000)'; % Large number to ensure correlation structure very distinct
y = zeros(length(t),nROIs);
% ROIs correlated and activated
for i = 1:nActive 
    y(:,i) = sin(t)+0.1*randn(size(t));
end
% ROIs correlated but not activated
for i = nActive+1:nActive+nNonActive 
    y(:,i) = cos(t)+0.1*randn(size(t));
end
% ROIs not correlated or activated
for i = nActive+nNonActive+1:nROIs
    y(:,i) = 0.1*randn(size(t));
end
Kint = inv(cov(y));

% [S,C] = graphconncomp(Kint,'direct','false');

% Generate regressor
protocol = zeros(nTpts,1);
win = 20; protocol(10:win:nTpts) = 1;
X = conv(protocol,genericHDR(1));
X = X(1:nTpts); X = X/max(X);
% Initialization
sigOLS = zeros(nThresh,nROIs,nIter);
sigGL = zeros(nThresh,nROIs,nIter);
sigOAS = zeros(nThresh,nROIs,nIter);
sigRidge = zeros(nThresh,nROIs,nIter);
for k = 1:nIter
    % Generate data
    tcRest = zeros(nTpts,nROIs,nSubs);
    tcTask = zeros(nTpts,nROIs,nSubs);
    for sub = 1:nSubs
        % Generate intrinsic network for each subject
%         KintSub = Kint+0.1*sprandsym(nROIs,dense,0.5,1);
        KintSub = Kint;
%         d(sub) = kullbackLeiblerDivergence(Kint,KintSub);
        % Generate resting-state data
        tcRest(:,:,sub) = randnorm(nTpts,zeros(nROIs,1),[],inv(gamma*KintSub))';
%         tcRest(:,:,sub) = tcRest(:,:,sub)+sqrt(ones(nTpts,1)*(std(tcRest(:,:,sub)).^2)/snrRS).*randn(nTpts,nROIs);
        % Generate task data
        beta = randnorm(1,[sig*ones(nActive,1)+1.5*randn(nActive,1);randn(nROIs-nActive,1)],[],inv(gamma*KintSub))';
        tcTask(:,:,sub) = X*beta+noise*randn(nTpts,nROIs);
    end
    % beta Estimation
    betaOLS = zeros(nSubs,nROIs);
    betaGL = zeros(nSubs,nROIs);
    betaOAS = zeros(nSubs,nROIs);
    betaRidge = zeros(nSubs,nROIs);
    for sub = 1:nSubs
        % Normalization
        tcRest(:,:,sub) = tcRest(:,:,sub)-ones(size(tcRest,1),1)*mean(tcRest(:,:,sub));
        tcRest(:,:,sub) = tcRest(:,:,sub)./(ones(size(tcRest,1),1)*std(tcRest(:,:,sub)));
        tcTask(:,:,sub) = tcTask(:,:,sub)-ones(size(tcTask,1),1)*mean(tcTask(:,:,sub));
        tcTask(:,:,sub) = tcTask(:,:,sub)./(ones(size(tcTask,1),1)*std(tcTask(:,:,sub)));
        
        % OLS
        Y = tcTask(:,:,sub);
        betaOLS(sub,:) = X\Y;
        
        % GL
        paramSelMethod = 1;
        optMethod = 1; % 1 = GL via two-metric projection, 2 = (Friedman,2007)
        kFolds = 2;
        nLevels = 5;
        nGridPts = 5;
        model = 8;
        modelEvid = @(V)modelEvidence(X,Y,V,model);
        K = sparseGGM(tcRest(:,:,sub),paramSelMethod,optMethod,'linear',kFolds,nLevels,nGridPts,modelEvid);
        betaGL(sub,:) = bayesianRegression(X,Y,K,model);
        
        % OAS covariance estimation
        K = inv(oas(tcRest(:,:,sub)));
        betaOAS(sub,:) = bayesianRegression(X,Y,K,model,1e-9);
        
        % Ridge
        betaRidge(sub,:) = bayesianRegression(X,Y,Id,model);
        disp(['Done subject',int2str(sub)]);
    end
    % Statistical inference
    [h,pOLS,ci,stat] = ttest(betaOLS);
    [h,pGL,ci,stat] = ttest(betaGL);
    [h,pOAS,ci,stat] = ttest(betaOAS);
    [h,pRidge,ci,stat] = ttest(betaRidge);
    for i = 1:nThresh
        sigOLS(i,:,k) = pOLS < thresh(i);
        sigGL(i,:,k) = pGL < thresh(i);
        sigOAS(i,:,k) = pOAS < thresh(i);
        sigRidge(i,:,k) = pRidge < thresh(i);
    end
end

% Plotting ROC   
figure; plot(mean(sum(sigOLS(:,nActive+1:nROIs,:),2),3)/(nROIs-nActive),mean(sum(sigOLS(:,1:nActive,:),2),3)/nActive,'k'); 
hold on; plot(mean(sum(sigGL(:,nActive+1:nROIs,:),2),3)/(nROIs-nActive),mean(sum(sigGL(:,1:nActive,:),2),3)/nActive,'b');
hold on; plot(mean(sum(sigOAS(:,nActive+1:nROIs,:),2),3)/(nROIs-nActive),mean(sum(sigOAS(:,1:nActive,:),2),3)/nActive,'r');
hold on; plot(mean(sum(sigRidge(:,nActive+1:nROIs,:),2),3)/(nROIs-nActive),mean(sum(sigRidge(:,1:nActive,:),2),3)/nActive,'g');

% Saving results
% save(['sigOLSsnr',int2str(snr*100),'Iter',int2str(nIter)],'sigOLS');
% save(['sigGLsnr',int2str(snr*100),'Iter',int2str(nIter)],'sigGL');
% save(['sigOASsnr',int2str(snr*100),'Iter',int2str(nIter)],'sigOAS');
% save(['sigRidgesnr',int2str(snr*100),'Iter',int2str(nIter)],'sigRidge');
% save(['sigRandsnr',int2str(snr*100),'Iter',int2str(nIter)],'sigRand');