% Learning Sparse Group Connectivity with Sparse Gaussian Graphical Model
% Input:    X = nxpxs observation matrix, n=#samples, p = #features, s = #subjects
%           nLevels = #levels for refining optimal lambda 
%           kFolds = #folds for learning optimal lambda
%           nGridPts = #grid points per refinement level
%           maxIter = maximum #iterations
% Output:   K = pxpxs subject-specific precision matrices
%           Cgrp = pxp sparse group covariance matrix
function [Cgrp,K] = sgggmCV(X,nLevels,kFolds,nGridPts,maxIter,Copt)
filepath = 'D:\research\';
addpath(genpath([filepath,'covarianceEstimationBN\']));
addpath(genpath([filepath,'toolboxes\quic\']));
% Initialization
[nSamp,nFeat,nSub] = size(X);
offDiag = ~eye(nFeat);
nu = nFeat+1; % Degrees of freedom for Wishart prior
K = zeros(nFeat,nFeat,nSub); % subject-specific precision
C = zeros(nFeat,nFeat,nSub); % subject-specific empirical covariance 
for s = 1:nSub
    [temp,C(:,:,s)] = oas(X(:,:,s));
    K(:,:,s) = temp^-1;
end
Cgrp = oas(mean(C,3)); % Group covariance
Ksum = sum(K,3);
lambda = max(abs(Ksum(offDiag)/nu))/10;
alpha = nu*nFeat/(Cgrp(:)'*Ksum(:)+nu*lambda*sum(abs(Cgrp(offDiag))));
S = nSamp*C; % scatter matrix
for n = 1:maxIter
    for s = 1:nSub
        K(:,:,s) = (nSamp+nu)*(S(:,:,s)+alpha*Cgrp)^-1; % Expected value of K
    end
    Ksum = sum(K,3);
%     lambdaMax = max(abs(Ksum(offDiag)/nu));
%     lambdaMin = lambdaMax/100;
%     lambdaBest = lambdaMax/10;
%     [trainInd,testInd] = cvSeq(nSamp,kFolds);
%     lambdaAcc = [];
%     for i = 1:nLevels
%         if i == 1
%             lambdaGrid = logspace(log10(lambdaMin),log10(lambdaMax),nGridPts);
%         else
%             if lambdaBest == lambdaGrid(1)
%                 lambdaGrid = logspace(log10(lambdaGrid(2)),log10(lambdaGrid(1)),nGridPts+2);
%             elseif lambdaBest == lambdaGrid(end)
%                 lambdaGrid = logspace(log10(lambdaGrid(end)/10),log10(lambdaGrid(end-1)),nGridPts+2);
%             else
%                 ind = find(lambdaGrid == lambdaBest);
%                 lambdaGrid = logspace(log10(lambdaGrid(ind+1)),log10(lambdaGrid(ind-1)),nGridPts+2);
%             end
%         end
%         lambdaGrid = fliplr(lambdaGrid);
%         evid = -inf*ones(length(lambdaGrid),kFolds);
%         skip = 0;
%         for j = 1:nGridPts
%             lambda = lambdaGrid(j);
%             if ((lambda ~= lambdaBest) && sum(lambdaAcc == lambda)) || (skip == 1)
%                 continue; % Avoid computing with the same lambda
%             end
%             lambdaAcc = [lambdaAcc;lambda]; % Stored computed lambdas
%             for k = 1:kFolds
%                 nTrain = length(trainInd{k});
%                 nTest = length(testInd{k});
%                 Xtrain = X(trainInd{k},:,:);
%                 Xtest = X(testInd{k},:,:);
%                 Ktrain = zeros(nFeat,nFeat);
%                 Ktest = zeros(nFeat,nFeat);
%                 for s = 1:nSub
%                     Ktrain = Ktrain + (nTrain+nu)*(Xtrain(:,:,s)'*Xtrain(:,:,s)+alpha*Cgrp)^-1;
%                     Ktest = Ktest + (nTest+nu)*(Xtest(:,:,s)'*Xtest(:,:,s)+alpha*Cgrp)^-1;
%                 end
%                 Ktrain = Ktrain/nu;
%                 Ktest = Ktest/nu;
%                 nIter = 200;
%                 [phi,~,~,~,iter] = quic('default',Ktrain,lambda*offDiag,1e-6,0,nIter);
% %                 sggmDualGap(phi,Ktrain,lambda*offDiag)
%                 if sggmDualGap(phi,Ktrain,lambda*offDiag) < 1e-6
% %                 if iter < nIter
%                     evid(j,k) = logDataLikelihood(Ktest,phi);
%                 else
%                     skip = 1;
%                     break;
%                 end
%             end
%         end
%         [~,ind] = max(mean(evid,2));
%         lambdaBest = lambdaGrid(ind);
%     end
    lambdaBest = 0;
    phi = quic('default',Ksum/nu,lambdaBest*offDiag,1e-9,0,1000);
    Cgrp = phi/alpha;
    alpha = nu*nFeat/(Cgrp(:)'*Ksum(:)+nu*lambdaBest*sum(abs(Cgrp(offDiag))));
    
%     nu = nSamp*trace(Cgrp*mean(K,3))/nFeat

    alpha
    lambdaBest
    max(abs(Ksum(offDiag)/nu))
    max(abs(Cgrp(:)))
    sum(Cgrp(offDiag)~=0)
    %%%%% Add Convergence criterion %%%%
    
end
    