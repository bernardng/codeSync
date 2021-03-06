% Simulation of Resting-state/Task data pairs
clear all; close all;
localpath = '/volatile/bernardng/';
netwpath = '/home/bn228083/';
addpath(genpath([localpath,'matlabToolboxes/lightspeed']));
addpath(genpath([localpath,'matlabToolboxes/markSchmidtCode']));
addpath(genpath([netwpath,'code/covarianceEstimationBN']));
addpath(genpath([netwpath,'code/bayesianRegressionBN']));
% Parameters
nSubs = 10; 
nROIs = 112; 
Id = eye(nROIs);
nActive = 20;
nNonActive = 20;
nTpts = 100;
dense = 0.05;
gamma = 10; 
sig = 1; 
snr = 0.2; 
noise = sqrt(sig^2/snr);
nIter = 5;
thresh = 0:0.05:1;
nThresh = length(thresh);
% Generate inverse covariance of intrinsic network
t = (1:50000)'; % Large number to ensure correlation structure very distinct
y = zeros(length(t),nROIs);
sigma = 0.001;
% ROIs correlated and activated
for i = 1:nActive 
    y(:,i) = sin(t)+sigma*randn(size(t));
end
% ROIs correlated but not activated
for i = nActive+1:nActive+nNonActive 
    y(:,i) = cos(t)+sigma*randn(size(t));
end
% ROIs not correlated or activated
for i = nActive+nNonActive+1:nROIs
    y(:,i) = sigma*randn(size(t));
end
Kint = inv(cov(y));

% % Add ROIs that are connected to rho*nROIs
% roi = nROIs:nROIs;
% rho = 0.25;
% conn = randperm(nROIs);
% conn = conn(1:round(rho*nROIs));
% KintMed = median(median(Kint(1:nActive,1:nActive)));
% Kint(roi,conn) = rand(length(roi),length(conn))*KintMed;
% Kint(conn,roi) = rand(length(conn),length(roi))*KintMed;
% Kint = (Kint+Kint')/2;
% % Make Kint diagonally dominant to ensure positive definite
% maxIter = 100; % To avoid infinite loop
% for i = 1:maxIter
%     Kint = Kint+eye(nROIs);
%     try
%         size(chol(Kint));
%     catch ME
%         disp('Increasing diagonal...');
%     end
% end
    
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
        % Generate task data
        beta = randnorm(1,[sig*ones(nActive,1)+1*randn(nActive,1);1*randn(nROIs-nActive,1)],[],inv(gamma*KintSub))';
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
        
%         % GL
%         paramSelMethod = 1; % 1 = CV, 2 = Model Evidence
%         optMethod = 1; % 1 = GL via two-metric projection, 2 = (Friedman,2007)
%         kFolds = 3;
%         nLevels = 5;
%         nGridPts = 5;
%         model = 1;
%         modelEvid = @(V)modelEvidence(X,Y,V,model);
%         K = sparseGGM(tcRest(:,:,sub),paramSelMethod,optMethod,'linear',kFolds,nLevels,nGridPts,modelEvid);
%         betaGL(sub,:) = bayesianRegression(X,Y,K,model);
        
        % OAS covariance estimation
        K = inv(oas(tcRest(:,:,sub)));
        betaOAS(sub,:) = bayesianRegression(X,Y,K);
        
%         % Ridge
%         betaRidge(sub,:) = bayesianRegression(X,Y,Id);
%         disp(['Done subject',int2str(sub)]);
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