% Synthetic data testing for sgggmCV
clear all; close all;
addpath(genpath('D:/research/covarianceEstimationBN'));

% Parameter Selection
nFeat = 100;
nSamp = 198; % Make sure divisible by 3
nSub = 60;
nIter = 5;
df = nFeat+1;
% df = 10*nFeat*(nFeat-1)/2;
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
    Kgrp = sprandsym(nFeat,0.2,0.9);
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
        C(:,:,s) = cov(X(:,:,s));
%         Coas(:,:,s) = oas(X(:,:,s));
        ClogEucl(:,:,s) = logm(C(:,:,s));
        Cinv(:,:,s) = inv(C(:,:,s));
    end
    
    % Compute Euclidean Mean
    Ceucl = mean(C,3); % Maybe use Coas??
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
    
    % Compute Frechet Mean
    
    % Compute SGGGM Mean
    nLevels = 3;
    kFolds = 3;
    nGridPts = 5;
    maxIter = 30;
    [Csgggm,Ksgggm,objAcc] = sgggmCV(X,nLevels,kFolds,nGridPts,maxIter);
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
% [h,p] = ttest(distEucl,distSGGGM);
