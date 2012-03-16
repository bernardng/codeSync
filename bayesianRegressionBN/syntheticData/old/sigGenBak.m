% Simulation of Resting-state/Task data pairs
clear all; close all;
filepath = 'D:/research/';
addpath(genpath([filepath,'toolboxes/l1General']));
addpath(genpath([filepath,'toolboxes/lightspeed']));
addpath(genpath([filepath,'toolboxes/minFunc']));
addpath(genpath([filepath,'covarianceEstimationNg']));
% Parameters
nSubs = 10; 
nROIs = 100; 
nTpts = 25;
dense = 0.05; 
gamma = 10; 
sig = 1; 
snr = 0.2; 
noise = sqrt(sig^2/snr);
nIter = 100;
nPerm = 5000;
thresh = 0:0.05:1;
nThresh = length(thresh);
method = 1;
% Generate inverse covariance of intrinsic network
nActive = 20;
nNonActive = 20;
t = (1:10000)';
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
sigROI = zeros(nThresh,nROIs,nIter);
sigGL = zeros(nThresh,nROIs,nIter);
sigOAS = zeros(nThresh,nROIs,nIter);
sigRidge = zeros(nThresh,nROIs,nIter);
sigRand = zeros(nThresh,nROIs,nIter);
for k = 1:nIter
    % Generate data
    tcRest = zeros(nTpts,nROIs,nSubs);
    tcTask = zeros(nTpts,nROIs,nSubs);
    for sub = 1:nSubs
        % Generate intrinsic network for each subject
        KintSub = Kint+0.1*sprandsym(nROIs,dense,0.5,1);
%         d(sub) = kullbackLeiblerDivergence(Kint,KintSub);
        % Generate resting-state data
        tcRest(:,:,sub) = randnorm(nTpts,zeros(nROIs,1),[],inv(gamma*KintSub))';
        % Generate task data
        beta = randnorm(1,[sig*ones(nActive,1)+1.5*randn(nActive,1);randn(nROIs-nActive,1)],[],inv(gamma*KintSub))';
        tcTask(:,:,sub) = X*beta+noise*randn(nTpts,nROIs);
    end
    % beta Estimation
    betaOLS = zeros(nSubs,nROIs);
    betaGL = zeros(nSubs,nROIs);
    betaOAS = zeros(nSubs,nROIs);
    betaRidge = zeros(nSubs,nROIs);
    betaRand = zeros(nSubs,nROIs);
    for sub = 1:nSubs
        % Normalization
        tcRest(:,:,sub) = tcRest(:,:,sub)-ones(size(tcRest,1),1)*mean(tcRest(:,:,sub));
        tcRest(:,:,sub) = tcRest(:,:,sub)./(ones(size(tcRest,1),1)*std(tcRest(:,:,sub)));
        tcTask(:,:,sub) = tcTask(:,:,sub)-ones(size(tcTask,1),1)*mean(tcTask(:,:,sub));
        tcTask(:,:,sub) = tcTask(:,:,sub)./(ones(size(tcTask,1),1)*std(tcTask(:,:,sub)));
        
        % OLS
        Y = tcTask(:,:,sub);
        betaOLS(sub,:) = X\Y;
        
        % Estimate sparse inverse covariance
        S = cov(tcRest(:,:,sub));
        lambdaList = [0.75,0.5,0.25];
        evid = zeros(length(lambdaList),1);
        L = zeros(nROIs,nROIs,length(lambdaList));
        for i = 1:length(lambdaList);
            lambda = lambdaList(i);
            nonZero = find(ones(nROIs));
            funObj = @(x)sparsePrecisionObj(x,nROIs,nonZero,S);
            Krest = eye(nROIs); options.order = -1;
            %     reg = lambda*(~eye(nROIs));
            reg = lambda*ones(nROIs);
            Krest(nonZero) = L1GeneralProjection(funObj,Krest(nonZero),reg(:),options);
            L(:,:,i) = Krest;
            % Extension of Tom Minka 2001 to have separate precision for Y and beta
            [dummy,evid(i)] = alphaEst(X,Y,L(:,:,i));
        end
        [dummy,ind] = max(evid);
        L = L(:,:,ind);
        alpha = alphaEst(X,Y,L);
        betaGL(sub,:) = ((eye(nROIs)+alpha*L)\(Y'*X)/(X'*X))';
        
        % OAS covariance estimation
        L = inv(oas(tcRest(:,:,sub)));
        alpha = alphaEst(X,Y,L);
        betaOAS(sub,:) = ((eye(nROIs)+alpha*L)\(Y'*X)/(X'*X))';
        
        % Ridge
        L = eye(nROIs); % Ridge
        alpha = alphaEst(X,Y,L);
        betaRidge(sub,:) = ((eye(nROIs)+alpha*L)\(Y'*X)/(X'*X))';
        
        % Random p.d.
        L = sprandsym(nROIs,dense,0.5,1);
        alpha = alphaEst(X,Y,L);
        betaRand(sub,:) = ((eye(nROIs)+alpha*L)\(Y'*X)/(X'*X))';
        
        %     f = @(beta)betaEstRegLS(beta,X,Y,L,alpha);
        %     beta0 = betaOLS(sub,:);
        %     betaReg(sub,:) = minFunc(f,beta0(:));
        %     betaReg(sub,:) = vbParamEst(Y,X,L);
        disp(['Done subject',int2str(sub)]);
    end
    [h,pOLS,ci,stat] = ttest(betaOLS);
    [h,pGL,ci,stat] = ttest(betaGL);
    [h,pOAS,ci,stat] = ttest(betaOAS);
    [h,pRidge,ci,stat] = ttest(betaRidge);
    [h,pRand,ci,stat] = ttest(betaRand);
    
    for i = 1:length(p)
        sigOLS(i,:,k) = pOLS < p(i);
        sigGL(i,:,k) = pGL < p(i);
        sigOAS(i,:,k) = pOAS < p(i);
        sigRidge(i,:,k) = pRidge < p(i);
        sigRand(i,:,k) = pRand < p(i);
    end
end

% Plotting ROC   
figure; plot(mean(sum(sigOLS(:,nActive+1:nROIs,:),2),3)/(nROIs-nActive),mean(sum(sigOLS(:,1:nActive,:),2),3)/nActive,'k'); 
hold on; plot(mean(sum(sigGL(:,nActive+1:nROIs,:),2),3)/(nROIs-nActive),mean(sum(sigGL(:,1:nActive,:),2),3)/nActive,'b');
hold on; plot(mean(sum(sigOAS(:,nActive+1:nROIs,:),2),3)/(nROIs-nActive),mean(sum(sigOAS(:,1:nActive,:),2),3)/nActive,'r');
hold on; plot(mean(sum(sigRidge(:,nActive+1:nROIs,:),2),3)/(nROIs-nActive),mean(sum(sigRidge(:,1:nActive,:),2),3)/nActive,'g');
hold on; plot(mean(sum(sigRand(:,nActive+1:nROIs,:),2),3)/(nROIs-nActive),mean(sum(sigRand(:,1:nActive,:),2),3)/nActive,'m');


% save(['sigOLSsnr',int2str(snr*100),'Iter',int2str(nIter)],'sigOLS');
% save(['sigGLsnr',int2str(snr*100),'Iter',int2str(nIter)],'sigGL');
% save(['sigOASsnr',int2str(snr*100),'Iter',int2str(nIter)],'sigOAS');
% save(['sigRidgesnr',int2str(snr*100),'Iter',int2str(nIter)],'sigRidge');
% save(['sigRandsnr',int2str(snr*100),'Iter',int2str(nIter)],'sigRand');