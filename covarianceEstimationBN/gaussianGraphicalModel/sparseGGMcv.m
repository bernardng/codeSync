% Sparse Gaussain graphical model via two-metric projection or (Friedman,2007)
% Input:    X = nxd feature matrix, n = #samples, d = #features
%           kFolds = #folds for cross validation
%           nLevels = #levels for choosing lambda 
%           nGridPts = #grid points per level
% Output:   K = dxd sparse inverse covariance matrix
%           lambdaBest = best lambda based on data likelihood
function [K,lambdaBest] = sparseGGMcv(X,kFolds,nLevels,nGridPts)
netwpath = '/home/bn228083/';
addpath(genpath([netwpath,'matlabToolboxes/markSchmidtCode']));
[n,d] = size(X);
S = cov(X);
lambdaMax = max(S(~eye(d)));
lambdaMin = lambdaMax/100;
lambdaBest = lambdaMax/10; % Initialization
[trainInd,testInd] = cvSeq(n,kFolds); % Create validation folds
lambdaAcc = []; % Skip computed lambda during refinement
% Refinement level
for i = 1:nLevels
    if i == 1
        lambdaGrid = logspace(log10(lambdaMin),log10(lambdaMax),nGridPts);
    else
        if lambdaBest == lambdaGrid(1)
            lambdaGrid = logspace(log10(lambdaGrid(2)),log10(lambdaGrid(1)),nGridPts+2);
        elseif lambdaBest == lambdaGrid(end)
            lambdaGrid = logspace(log10(lambdaGrid(end)/10),log10(lambdaGrid(end-1)),nGridPts+2);
        else
            ind = find(lambdaGrid == lambdaBest);
            lambdaGrid = logspace(log10(lambdaGrid(ind+1)),log10(lambdaGrid(ind-1)),nGridPts+2);
        end
    end
    lambdaGrid = fliplr(lambdaGrid);
    % Compute K for each grid point
    evid = -inf*ones(length(lambdaGrid),kFolds);
    skip = 0; % Skip lower lambda if did not converge with current lambda
    for j = 1:nGridPts
        lambda = lambdaGrid(j);
        if ((lambda ~= lambdaBest) && sum(lambdaAcc == lambda)) || (skip == 1)
            continue; % Avoid computing with the same lambda
        end
        lambdaAcc = [lambdaAcc;lambda]; % Stored computed lambdas
        for k = 1:kFolds
            Xtrain = X(trainInd{k},:);
            Xtest = X(testInd{k},:);
            Strain = cov(Xtrain);
            Stest = cov(Xtest);

%             nonZero = find(ones(d));
%             funObj = @(x)sparsePrecisionObj(x,d,nonZero,Strain);
%             %%%%%% Add in warm start later using Kbest %%%%%%
%             K = eye(size(Strain));
%             options.order = -1; % LBFGS
%             options.maxIter = 100; % Default
%             reg = lambda*~eye(d);
%             % Estimate sparse inverse covariance
% %             [K(nonZero),converged] = L1GeneralProjection(funObj,K(nonZero),reg(:),options);
%             [K(nonZero),fEvals] = L1GeneralProjectionBN(funObj,K(nonZero),Strain,reg(:),options);
%             if fEvals < options.maxIter
%                 % Compute data likelihood
%                 evid(j,k) = logDataLikelihood(Stest,K);
%             end
            
            K = quic(Strain,lambda*~eye(size(Strain)));
            estDualGap(K,Strain,lambda*~eye(size(Strain)))
            if estDualGap(K,Strain,lambda*~eye(size(Strain))) < 1e-5
                evid(j,k) = logDataLikelihood(Stest,K);
            else
                skip = 1;
                break;
            end
        end
    end
    [dummy,ind] = max(mean(evid,2));
    lambdaBest = lambdaGrid(ind);
end
% Compute sparse inverse covariance using optimal lambda
% nonZero = find(ones(d));
% funObj = @(x)sparsePrecisionObjBN(x,d,nonZero,S);
% options.order = -1; % LBFGS
% reg = lambdaBest*~eye(d); % Penalize only off diagonal elements
% K = eye(size(S)); % Initialization
% K(nonZero) = L1GeneralProjection(funObj,K(nonZero),reg(:),options);
% K(nonZero) = L1GeneralProjection(funObj,K(nonZero),S,reg(:),options);
lambdaBest
K = quic(S,lambdaBest*~eye(size(S)));



        
        
        
        

