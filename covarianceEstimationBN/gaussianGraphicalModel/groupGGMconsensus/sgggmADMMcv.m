% Learning Sparse Group Connectivity with Sparse Gaussian Graphical Model
% Input:    X = nxpxs observation matrix, n=#samples, p = #features, s = #subjects
%           nLevels = #levels for refining optimal lambda 
%           kFolds = #folds for learning optimal lambda
%           nGridPts = #grid points per refinement level
%           maxIter = max # of iterations for each sgggm estimation
% Output:   Kgrp = pxp sparse group precision matrix
%           lambdaBest = optimal sparsity level
function [Kgrp,lambdaBest] = sgggmADMMcv(X,nLevels,kFolds,nGridPts,maxIter)
filepath = 'D:\research\';
addpath(genpath([filepath,'covarianceEstimationBN\']));
[n,d,s] = size(X);
offDiag = ~eye(d);
S = zeros(d,d,s);
for sub = 1:s
    S(:,:,sub) = cov(X(:,:,sub));
end
Sgrp = mean(S,3);
lambdaMax = max(abs(Sgrp(offDiag)));
scaleMax = 1;
scaleMin = 0.01;
scaleBest = 0.1; % Initialization
scaleAcc = scaleBest; % Skip computed lambda during refinement
% Create validation folds
[trainInd,testInd] = cvSeq(n,kFolds); 
nTrain = length(trainInd{1});
Xtrain = zeros(nTrain,d,s,kFolds);
Stest = zeros(d,d,s,kFolds);
for k = 1:kFolds
    Xtrain(:,:,:,k) = X(trainInd{k},:,:);
    Xtest = X(testInd{k},:,:);
    for sub = 1:s
        Stest(:,:,sub,k) = cov(Xtest(:,:,sub));
    end
end
matlabpoolUsed = 0;
try
    matlabpool('3');
catch ME
    matlabpoolUsed = 1;
end
% Refinement level
for i = 1:nLevels
    disp(['Refinement level ',int2str(i)]);
    if i == 1
        scaleGrid = logspace(log10(scaleMin),log10(scaleMax),nGridPts);
    else
        if abs(scaleBest-scaleGridMod(1))<1e-12
            scaleGrid = logspace(log10(scaleGridMod(2)),log10(scaleGridMod(1)*(10^(1/(2*i)))),nGridPts+1);
        elseif abs(scaleBest-scaleGridMod(end))<1e-12
            scaleGrid = logspace(log10(scaleGridMod(end)/(10^(1/(2*i)))),log10(scaleGridMod(end-1)),nGridPts+1);
        else
            ind = find(abs(scaleGridMod-scaleBest)<1e-12);
            scaleGrid = logspace(log10(scaleGridMod(ind+1)),log10(scaleGridMod(ind-1)),nGridPts+2);
        end
    end
    scaleGrid = fliplr(scaleGrid); % Always in descending order
    [dummy,ind,dummy] = find(abs(ones(length(scaleAcc),1)*scaleGrid-scaleAcc'*ones(1,length(scaleGrid)))<1e-12); % More robust than using set functions
    scaleGridMod = sort([scaleGrid(setdiff(1:length(scaleGrid),ind)),scaleBest],2,'descend'); % Remove computed scales
    scaleAcc = [scaleAcc,scaleGridMod]; % Store computed scales
    evid = -inf*ones(length(scaleGridMod),kFolds);
    % Cross validation to set sparsity level
    for k = 1:kFolds
        for j = 1:length(scaleGridMod)
            disp(['Grid point ',int2str(j)]);
            [KgrpTrain,converged] = sgggmADMM(Xtrain(:,:,:,k),lambdaMax*scaleGridMod(j),maxIter);
            if converged
                logDL = 0;
                for sub = 1:s                
                    logDL = logDL+logDataLikelihood(Stest(:,:,sub,k),KgrpTrain);
                end
                evid(j,k) = logDL;
            else
                break;
            end
        end
    end
    [dummy,ind] = max(mean(evid,2));
    scaleBest = scaleGridMod(ind);
end
if matlabpoolUsed == 0
    matlabpool('close');
end
lambdaBest = lambdaMax*scaleBest;
Kgrp = sgggmADMM(X,lambdaBest);

