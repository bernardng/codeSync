% Synthetic data testing for sgggmCV
clear all; close all;
nSamp = 100000;
nFeat = 100;
% Y = 0.01*randn(nSamp,nFeat);
% for i = 1:5
%     Y(:,i) = sin(1:nSamp)'+0.01*randn(nSamp,1);
% end
% Cgrp = corr(Y);
Kgrp = sprandsym(nFeat,0.2,0.9);
Kgrp = Kgrp+5*eye(nFeat);
Kgrp = corrcov(Kgrp);
nSub = 60;
% df = nFeat+1;
df = 10*nFeat*(nFeat-1)/2;
% Generating precision matrices
K = zeros(nFeat,nFeat,nSub);
for s = 1:nSub
    K(:,:,s) = wishrnd(Kgrp,df)/df;
end
% Generating timecourses
nSamp = 999;
X = zeros(nSamp,nFeat,nSub);
for s = 1:nSub
    X(:,:,s) = mvnrnd(zeros(1,nFeat),K(:,:,s)^-1,nSamp);
    % Normalization
    X(:,:,s) = X(:,:,s)-ones(nSamp,1)*mean(X(:,:,s));
    X(:,:,s) = X(:,:,s)./(ones(nSamp,1)*std(X(:,:,s)));
end
nLevels = 3;
kFolds = 3;
nGridPts = 5;
maxIter = 5;
[CgrpEst,Kest] = sgggmCV(X,nLevels,kFolds,nGridPts,maxIter);
% Convert estimated covariance to correlation
KgrpEst = inv(CgrpEst);
dia = diag(1./sqrt(KgrpEst(eye(nFeat)==1)));
Kpc = dia*KgrpEst*dia; % Why not to flip sign???
Kpc(eye(nFeat)==1) = 1;
