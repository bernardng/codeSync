% Weighted sparse Gaussain graphical model via QUIC (C.J. Hsieh et al., NIP2011)
% Input:    X = nxd feature matrix, n = #samples, d = #features
%           kFolds = #folds for cross validation
%           nLevels = #levels for choosing lambda 
%           nGridPts = #grid points per level
%           weight = dxd weights on |Kij|, currently set to exp(-weight/sigma) 
%           nWt = #grid points for sigma
% Output:   K = dxd sparse inverse covariance matrix
%           lambdaBest = best lambda based on data likelihood
%           sigmaBest = best sigma based on data likelihood
function [K,lambdaBest,sigmaBest] = wsggmCV(X,kFolds,nLevels,nGridPts,weight,nWt)
addpath(genpath('/home/bn228083/matlabToolboxes/quic'));
[n,d] = size(X);
S = cov(X);
lambdaMax = max(abs(S(~eye(d))));
scaleMax = 1;
scaleMin = 0.01;
scaleBest = 0.1; % Initialization
[trainInd,testInd] = cvSeq(n,kFolds); % Create validation folds
scaleAcc = scaleBest; % Skip computed lambda during refinement
% Refinement level
matlabpool('5');
for i = 1:nLevels
    if i == 1
        scaleGrid = logspace(log10(scaleMin),log10(scaleMax),nGridPts);
    else
        if abs(scaleBest-scaleGridMod(1))<1e-12
            scaleGrid = logspace(log10(scaleGridMod(2)),log10(scaleGridMod(1)/(10^(1/(2*i)))),nGridPts+1);
        elseif abs(scaleBest-scaleGridMod(end))<1e-12
            scaleGrid = logspace(log10(scaleGridMod(end)/(10^(1/(2*i)))),log10(scaleGridMod(end-1)),nGridPts+1);
        else
            ind = find(abs(scaleGridMod-scaleBest)<1e-12);
            scaleGrid = logspace(log10(scaleGridMod(ind+1)),log10(scaleGridMod(ind-1)),nGridPts+2);
        end
    end
    scaleGrid = fliplr(scaleGrid); % Always in descending order
    [~,ind,~] = find(abs(ones(length(scaleAcc),1)*scaleGrid-scaleAcc'*ones(1,length(scaleGrid)))<1e-12); % More robust than using set functions
    scaleGridMod = sort([scaleGrid(setdiff(1:length(scaleGrid),ind)),scaleBest],2,'descend'); % Remove computed scales
    scaleAcc = [scaleAcc,scaleGridMod]; % Store computed scales
    mu = mean(weight(weight>0)); % Fiber counts appear exponentially distributed
    lb = prctile(weight(weight>0),5);
    ub = prctile(weight(weight>0),95);
    sigmaGrid = linspace(lb,ub,nWt); % Between 25th and 75th percentile
    evid = -inf*ones(length(scaleGridMod),nWt,kFolds);
    % Cross validation to set sparsity level
    for k = 1:kFolds
        Xtrain = X(trainInd{k},:);
        Xtest = X(testInd{k},:);
        Strain = cov(Xtrain);
        Stest = cov(Xtest);
        for j = 1:length(scaleGridMod)
            %K = QUIC('path',Strain,lambdaMax.*scaleGridMod(j).*~eye(d),exp(-weight/sigmaGrid),1e-9,2,200);
    	    parfor w = 1:nWt
		K(:,:,w) = QUIC('default',Strain,lambdaMax.*scaleGridMod(j).*exp(-weight/sigmaGrid(w)).*~eye(d),1e-9,2,200);
                dg(w) = dualGap(K(:,:,w),Strain,lambdaMax.*scaleGridMod(j).*exp(-weight/sigmaGrid(w)).*~eye(d))
		if dg(w) < 1e-5                
		    evid(j,w,k) = logDataLikelihood(Stest,K(:,:,w));
		end
	    end
            if sum(dg < 1e-5)==0
		break;
            end
        end
    end
    evidAve = mean(evid,3);
    evidMax = max(evidAve(:));
    ind = find(evidAve==evidMax);
    [x,y] = ind2sub(size(evidAve),ind);
    xMax = max(x);
    y = max(y(x==xMax));
    x = xMax;
    scaleBest = scaleGridMod(x);
    sigmaBest = sigmaGrid(y);
end
matlabpool('close');
% Compute sparse inverse covariance using optimal lambda
lambdaBest = lambdaMax*scaleBest;
K = QUIC('default',S,lambdaBest.*exp(-weight/sigmaBest).*~eye(d),1e-9,2,200);
