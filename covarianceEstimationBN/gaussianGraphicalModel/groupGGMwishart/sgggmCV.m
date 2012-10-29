% Learning Sparse Group Connectivity with Sparse Gaussian Graphical Model
% Input:    X = nxpxs observation matrix, n=#samples, p = #features, s = #subjects
%           nLevels = #levels for refining optimal lambda 
%           kFolds = #folds for learning optimal lambda
%           nGridPts = #grid points per refinement level
%           maxIter = maximum #iterations
% Output:   K = pxpxs subject-specific precision matrices
%           Cgrp = pxp sparse group covariance matrix
function [Cgrp,K] = sgggmCV(X,nLevels,kFolds,nGridPts,maxIter)
filepath = 'D:\research\';
addpath(genpath([filepath,'covarianceEstimationBN\']));
addpath(genpath([filepath,'toolboxes\quic\']));
% Initialization
[nSamp,nFeat,nSub] = size(X);
offDiag = ~eye(nFeat);
% nu = nFeat+1; % Degrees of freedom for Wishart prior
K = zeros(nFeat,nFeat,nSub); % subject-specific precision
C = zeros(nFeat,nFeat,nSub); % subject-specific empirical covariance 
for s = 1:nSub
    [Coas,C(:,:,s)] = oas(X(:,:,s));
    K(:,:,s) = Coas^-1;
end
S = nSamp*C;
Ktrain = zeros(nFeat,nFeat,nSub,kFolds);
Ktest = zeros(nFeat,nFeat,nSub,kFolds);
Strain = zeros(nFeat,nFeat,nSub,kFolds);
Stest = zeros(nFeat,nFeat,nSub,kFolds);
[trainInd,testInd] = cvSeq(nSamp,kFolds); % Create validation folds
for k = 1:kFolds
    for s = 1:nSub
        [CtrainOAS,Ctrain] = oas(X(trainInd{k},:,s));
        Ktrain(:,:,s,k) = CtrainOAS^-1;
        Strain(:,:,s,k) = length(trainInd{k})*Ctrain;
        [CtestOAS,Ctest] = oas(X(testInd{k},:,s));
        Ktest(:,:,s,k) = CtestOAS^-1;
        Stest(:,:,s,k) = length(testInd{k})*Ctest;
    end
end

for n = 1:maxIter
    Kgrp = sum(K,3)/((nFeat+1)*nSub);
    
    % Refined grid for choosing lambda
    lambdaMax = max(abs(Kgrp(offDiag)));
    scaleMax = 1;
    scaleMin = 0.01;
    scaleBest = 0.1; % Initialization
    
    scaleAcc = scaleBest; % Skip computed lambda during refinement
    % Refinement level
    for i = 1:nLevels
        if i == 1
            scaleGrid = logspace(log10(scaleMin),log10(scaleMax),nGridPts);
        else
            if abs(scaleBest-scaleGrid(1))<1e-6
                scaleGrid = logspace(log10(scaleGrid(2)),log10(scaleGrid(1)),nGridPts+1);
            elseif abs(scaleBest-scaleGrid(end))<1e-6
                scaleGrid = logspace(log10(scaleGrid(end)/10),log10(scaleGrid(end-1)),nGridPts+1);
            else
                ind = find(abs(scaleGrid-scaleBest)<1e-6);
                scaleGrid = logspace(log10(scaleGrid(ind+1)),log10(scaleGrid(ind-1)),nGridPts+2);
            end
        end
        scaleGrid = fliplr(scaleGrid); % Always in descending order
        [~,ind,~] = find(abs(ones(length(scaleAcc),1)*scaleGrid-scaleAcc'*ones(1,length(scaleGrid)))<1e-6); % More robust than using set functions
        scaleGridMod = sort([scaleGrid(setdiff(1:length(scaleGrid),ind)),scaleBest],2,'descend'); % Remove computed scales
        scaleAcc = [scaleAcc,scaleGridMod]; % Store computed scales
        evid = -inf*ones(length(scaleGridMod),kFolds);
        % Cross validation to set sparsity level
        for k = 1:kFolds
            KgrpTrain = sum(Ktrain(:,:,:,k),3)/((nFeat+1)*nSub);
            KgrpTest = sum(Ktest(:,:,:,k),3)/((nFeat+1)*nSub);
            for j = 1:length(scaleGridMod)
                if j == 1
                    CgrpTrain = QUIC('default',KgrpTrain,lambdaMax*scaleGridMod(j)*offDiag,1e-9,2,200);
                else
                    CgrpTrain = QUIC('default',KgrpTrain,lambdaMax*scaleGridMod(j)*offDiag,1e-9,2,200,CgrpTrain,inv(CgrpTrain));
                end
                dg = dualGap(CgrpTrain,KgrpTrain,lambdaMax*scaleGridMod(j)*offDiag)
                % Check convergence
                if dg < 1e-5
                    evid(j,k) = logDataLikelihood(KgrpTest,CgrpTrain);
                else
                    break;
                end
            end
        end
        [dummy,ind] = max(mean(evid,2));
        scaleBest = scaleGridMod(ind);
    end
    lambdaBest = lambdaMax*scaleBest;
    Cgrp = QUIC('default',Kgrp,lambdaBest*offDiag,1e-9,0,200);
    
    for s = 1:nSub
        K(:,:,s) = nSamp*(S(:,:,s)+Cgrp)^-1; % Expected value of K
    end
    
    for k = 1:kFolds
        for s = 1:nSub
            Ktrain(:,:,s,k) = length(trainInd{k})*(Strain(:,:,s,k)+Cgrp)^-1;
            Ktest(:,:,s,k) = length(testInd{k})*(Stest(:,:,s,k)+Cgrp)^-1;
        end
    end
    
    %%%%% Add Convergence criterion %%%%
    
end
    