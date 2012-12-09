% Synthetic data testing for sgggmCV
clear all; 
% close all;
addpath(genpath('D:/research/covarianceEstimationBN'));
addpath(genpath('D:/research/toolboxes/general'));

% Parameter Selection
nFeat = 100;
nSamp = 30; % Make sure divisible by 3
nSub = 60;
nIter = 1;
% df = nFeat+1;
df = 10*nFeat*(nFeat-1)/2;
lTri = tril(ones(nFeat),-1)==1;

distEucl = zeros(nIter,1);
distEuclSub = zeros(nIter,1);
distLogEucl = zeros(nIter,1);
% distFrechet = zeros(nIter,1);
distSGGGM = zeros(nIter,1);
distSGGGMsub = zeros(nIter,1);
matlabpoolUsed = 0;
try
    matlabpool('3');
catch ME
    matlabpoolUsed = 1;
end
for n = 1:nIter
    % Generate group precision
    Kgrp = sprandsym(nFeat,0.5,0.9);
    Kgrp = Kgrp+5*eye(nFeat);
    Kgrp = corrcov(Kgrp);
    KgrpSqrt = Kgrp^(0.5);
    
    % Generating subject precision matrices
    K = zeros(nFeat,nFeat,nSub);
    Csqrt = zeros(nFeat,nFeat,nSub);
    for s = 1:nSub
        K(:,:,s) = wishrnd(Kgrp,df)/df;
        Csqrt(:,:,s) = K(:,:,s)^(-0.5);
    end
    
    % Generating timecourses and compute subject correlation matrices
    X = zeros(nSamp,nFeat,nSub);
    C = zeros(nFeat,nFeat,nSub);
    Coas = zeros(nFeat,nFeat,nSub);
    ClogEucl = zeros(nFeat,nFeat,nSub);
    Cinv = zeros(nFeat,nFeat,nSub);
    for s = 1:nSub
        X(:,:,s) = mvnrnd(zeros(1,nFeat),K(:,:,s)^-1,nSamp);
        % Normalization
        X(:,:,s) = X(:,:,s)-ones(nSamp,1)*mean(X(:,:,s));
        X(:,:,s) = X(:,:,s)./(ones(nSamp,1)*std(X(:,:,s)));
        % Compute subject correlation matrices
%         C(:,:,s) = cov(X(:,:,s));
        C(:,:,s) = oas(X(:,:,s));
        ClogEucl(:,:,s) = logm(C(:,:,s));
        Cinv(:,:,s) = inv(C(:,:,s));
    end
    
    if 1
        
    % Compute Euclidean Mean
    tc = [];
    for sub = 1:nSub
        tc = [tc;X(:,:,sub)];
    end
    tc = tc-ones(size(tc,1),1)*mean(tc);
    tc = tc./(ones(size(tc,1),1)*std(tc));
    Ceucl = oas(tc);
%     Ceucl = mean(C,3); % Maybe use Coas??
%     dia = diag(1./sqrt(Ceucl(eye(nFeat)==1)));
%     Ceucl = dia*Ceucl*dia;
    distTemp = logm(KgrpSqrt*Ceucl*KgrpSqrt);
    distEucl(n) = norm(distTemp(lTri));
    distTemp = 0;
    for s = 1:nSub
        temp = logm(Csqrt(:,:,s)*Cinv(:,:,s)*Csqrt(:,:,s));
        distTemp = distTemp+norm(temp(lTri));
    end
    distEuclSub(n) = distTemp/nSub;
    
    % Compute Log Euclidean Mean
    ClogEucl = expm(mean(ClogEucl,3));
%     dia = diag(1./sqrt(ClogEucl(eye(nFeat)==1)));
%     ClogEucl = dia*ClogEucl*dia;
    distTemp = logm(KgrpSqrt*ClogEucl*KgrpSqrt);
    distLogEucl(n) = norm(distTemp(lTri));
    
    end
    
    % Compute SGGGM Mean
    nLevels = 3;
    kFolds = 3;
    nGridPts = 5;
    maxIter = 30;
    
    nBootstrap = 50;
    Csgggm = zeros(nFeat,nFeat,nBootstrap);
    for b = 1:nBootstrap
        % Generate subsamples
        nSubsamp = 100;
%         nSubPerSamp = 30;
        Xall = [];
        for s = 1:nSubsamp
            Xacc = [];
%             ind = randperm(nSub);
%             for sub = 1:nSubPerSamp
%                 Xacc = [Xacc;X(:,:,ind(sub))];
%             end
            ind = randperm(nSamp);
            ind = ind(1:round(nSamp*2/4));
%             ind = randperm(10000);
%             ind = mod(ind,nSamp);
%             ind(ind==0) = nSamp;
%             ind = ind(1:nSamp/2);
            for sub = 1:nSub
                Xacc = [Xacc;X(ind,:,sub)];
            end
            Xacc = Xacc-ones(size(Xacc,1),1)*mean(Xacc);
            Xacc = Xacc./(ones(size(Xacc,1),1)*std(Xacc));
            Xall = cat(3,Xall,Xacc);
        end
        [Csgggm(:,:,b),Ksgggm,objAcc] = sgggmCV(Xall,nLevels,kFolds,nGridPts,maxIter);
    end
    Csgggm = mean(Csgggm,3);
    ind = randperm(nFeat);

%     [Csgggm,Ksgggm,objAcc] = sgggmCV(X,nLevels,kFolds,nGridPts,maxIter);
%     dia = diag(1./sqrt(Csgggm(eye(nFeat)==1)));
%     Csgggm = dia*Csgggm*dia; 
    distTemp = logm(KgrpSqrt*Csgggm*KgrpSqrt);
    distSGGGM(n) = norm(distTemp(lTri));
    distTemp = 0;
    for s = 1:nSub
        temp = logm(Csqrt(:,:,s)*Ksgggm(:,:,s)*Csqrt(:,:,s));
        distTemp = distTemp+norm(temp(lTri));
    end
    distSGGGMsub(n) = distTemp/nSub;
end
if matlabpoolUsed == 0
    matlabpool('close');
end

bar([mean(distEucl),mean(distLogEucl),mean(distSGGGM)]);
mean(distSGGGM)/mean(distEucl)
% [h,p] = ttest(distEucl,distSGGGM);
