% Synthetic data testing for sgggmCV
clear all; close all;
nSamp = 100000;
nFeat = 50;
% Y = 0.01*randn(nSamp,nFeat);
% for i = 1:5
%     Y(:,i) = sin(1:nSamp)'+0.01*randn(nSamp,1);
% end
% Cgrp = corr(Y);
Cgrp = sprandsym(nFeat,0.2,0.9);
Cgrp = Cgrp+2*eye(nFeat);
Cgrp = corrcov(Cgrp);
nSub = 60;
df = nFeat+1;
df = 100*nFeat*nFeat;
K = zeros(nFeat,nFeat,nSub);
for s = 1:nSub
    K(:,:,s) = wishrnd(Cgrp^-1,df)/df;
end
% nSamp = 186;
nSamp = 33;
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
[Cest,Kest] = sgggmCV(X,nLevels,kFolds,nGridPts,maxIter,Cgrp);
% Cest = corrcov(Cest);
