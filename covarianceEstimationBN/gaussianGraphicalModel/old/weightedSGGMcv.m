% Sparse Gaussain graphical model for integrating Kanat and Kfunc
% Input:    X = nxd feature matrix, n = #samples, d = #features
%           kFolds = #folds for cross validation
%           nLevels = #levels for choosing lambda 
%           nGridPts = #grid points per level
%           weight = dxd weights on |Kij|
% Output:   K = dxd sparse inverse covariance matrix
%           lambdaBest = best lambda based on data likelihood
%           sigmaBest = best sigma based on data likelihood
function [K,lambdaBest,sigmaBest] = weightedSGGMcv(X,kFolds,nLevels,nGridPts,weight,nWt)
netwpath = '/home/bn228083/';
% addpath(genpath([netwpath,'matlabToolboxes/markSchmidtCode']));
addpath(genpath([netwpath,'matlabToolboxes/quic']));
[n,d] = size(X);
S = cov(X);
lambdaMax = max(S(~eye(d)));
lambdaMin = lambdaMax/100;
lambdaBest = lambdaMax/10; % Initialization
[trainInd,testInd] = cvSeq(n,kFolds); % Create validation folds
lambdaAcc = []; % Skip computed lambda during refinement
% Refinement level
for i = 1:nLevels
    disp(['Refinement Level',int2str(i)]);
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
%     mu = mean(weight(weight>0)); % Fiber counts appear exponentially distributed
%     sigmaGrid = linspace(log(4/3)*mu,log(4)*mu,nWt); % Between 25th and 75th percentile
    
    sigmaGrid = linspace(0.5,8,nWt); % For binarized Kanat
    
    % Compute K for each grid point
    evid = -inf*ones(length(lambdaGrid),nWt,kFolds);
    skip = 0; % Skip lower lambda if did not converge with current lambda
    for j = 1:nGridPts
        disp(['Grid point',int2str(j)]);
        lambda = lambdaGrid(j);
        if ((lambda ~= lambdaBest) && sum(lambdaAcc == lambda)) || (skip == 1)
            continue; % Avoid computing with the same lambda
        end
        lambdaAcc = [lambdaAcc;lambda]; % Stored computed lambdas
        for w = 1:nWt
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
                
                K = quic(Strain,lambda.*exp(-weight/sigmaGrid(w)).*~eye(size(Strain)));
                estDualGap(K,Strain,lambda.*exp(-weight/sigmaGrid(w)).*~eye(size(Strain)))
                if estDualGap(K,Strain,lambda.*exp(-weight/sigmaGrid(w)).*~eye(size(Strain))) < 1e-5
                    evid(j,w,k) = logDataLikelihood(Stest,K);
                end
            end
        end
        for k = 1:kFolds % If for any fold, evid = -inf for all sigma, then skip subsequent lambda's
            if sum(isinf(evid(j,:,k))) == nWt
                skip = 1;
                break;
            end
        end
    end
    evidAve = mean(evid,3);
    [dummy,ind] = max(evidAve(:));
    [x,y] = ind2sub(size(evidAve),ind);
    lambdaBest = lambdaGrid(x);
    sigmaBest = sigmaGrid(y);
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
sigmaBest
K = quic(S,lambdaBest.*exp(-weight/sigmaBest).*~eye(size(S)));


        
        
        
        

