% Sparse Gaussain graphical model via QUIC (C.J. Hsieh et al., NIP2011)
% Input:    X = nxd feature matrix, n = #samples, d = #features
%           kFolds = #folds for cross validation
%           nLevels = #levels for choosing lambda 
%           nGridPts = #grid points per level
% Output:   K = dxd sparse inverse covariance matrix
%           lambdaBest = best lambda based on data likelihood
% Notes:    Please add path to QUIC
function [K,lambdaBest] = sggmCV(X,kFolds,nLevels,nGridPts)
addpath(genpath('/home/bn228083/matlabToolboxes/quic/'));
[n,d] = size(X);
S = cov(X);
lambdaMax = max(abs(S(~eye(d)))); % lambda at which all off diagonal elements are shrunk to zero
scaleMax = 1;
scaleMin = 0.01;
scaleBest = 0.1; % Initialization
[trainInd,testInd] = cvSeq(n,kFolds); % Create validation folds
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
        Xtrain = X(trainInd{k},:);
        Xtest = X(testInd{k},:);
        Strain = cov(Xtrain);
        Stest = cov(Xtest);
        % K = QUIC('path',Strain,lambdaMax*~eye(d),scaleGridMod,1e-9,2,200);
        for j = 1:length(scaleGridMod)
	    if j == 1	    
		K = QUIC('path',Strain,lambdaMax*scaleGridMod(j)*~eye(d),1e-9,2,200);
	    else
		K = QUIC('path',Strain,lambdaMax*scaleGridMod(j)*~eye(d),1e-9,2,200,K,inv(K));
	    end
	    dg = dualGap(K(:,:,j),Strain,lambdaMax*scaleGridMod(j)*~eye(d))
            % Check convergence
            if dg < 1e-5
                evid(j,k) = logDataLikelihood(Stest,K(:,:,j));
            else
                break;
            end
        end
    end
    [dummy,ind] = max(mean(evid,2));
    scaleBest = scaleGridMod(ind);
end
% Compute sparse inverse covariance using optimal lambda
lambdaBest = lambdaMax*scaleBest;
K = QUIC('default',S,lambdaBest*~eye(d),1e-9,2,200);
        
        
        
        

