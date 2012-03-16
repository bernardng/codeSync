% Sparse Gaussain graphical model via two-metric projection or (Friedman,2007)
% Input:    X = nxd feature matrix, n = #samples, d = #features
%           paramSelMethod = 1:cross validation, 2:model evidence
%           optMethod = 1:two-metric projection, 2: (Friedman,2007)
%           param = 'log' scale or user-specified lambda range
%           kFolds = #folds for cross validation
%           nLevels = #levels for choosing lambda 
%           nGridPts = #grid points per level
%           modelEvid = function handle of user-defined model evidence
%           varargin = variables required for computing user-defined model evidence
% Output:   K = dxd sparse inverse covariance matrix
function K = sparseGGM(X,paramSelMethod,optMethod,param,kFolds,nLevels,nGridPts,modelEvid,varargin)
netwpath = '/home/bn228083/';
addpath(genpath([netwpath,'matlabToolboxes/markSchmidtCode']));
[n,d] = size(X);
S = cov(X);
scale = 1; % To increase number of off diagonal elements
lambdaMax = max(S(~eye(d)))/scale;
% Determine optimal amount of regularization
if paramSelMethod == 1 % Cross validation 
    [trainInd,testInd] = cvSeq(n,kFolds);
    lambda = zeros(kFolds,1);
    evid = zeros(kFolds,1);
    Xsplit = cell(2,1);
    for fold = 1:kFolds
        Xsplit{1} = X(trainInd{fold},:);
        Xsplit{2} = X(testInd{fold},:);
        [lambda(fold),evid(fold)] = paramSel(Xsplit,optMethod,param,lambdaMax,nLevels,nGridPts);
    end
%     lambdaBest = evid'*lambda/sum(evid) % Weighted average lambda
    
    % Take lambda providing highest model evidence
    [dummy,ind] = max(evid);
    lambdaBest = lambda(ind);    
    
    % Compute sparse inverse covariance using optimal lambda
    S = cov(X); % Empirical covariance with all samples
    if optMethod == 1
        nonZero = find(ones(d));
        funObj = @(x)sparsePrecisionObjBN(x,d,nonZero,S);
        options.order = -1; % LBFGS
        reg = lambdaBest*~eye(d); % Penalize only off diagonal elements
        K = inv(oas(S)); % Initialization
        K(nonZero) = L1GeneralProjectionBN(funObj,K(nonZero),S,reg(:),options);
    else
        K = L1precisionBCD(S,lambdaBest);
    end
elseif paramSelMethod == 2 % Model evidence
    [lambdaBest,evidBest,K] = paramSel(X,optMethod,param,lambdaMax,nLevels,nGridPts,modelEvid,varargin{:});
    lambdaBest
end

    
    
        
        
        
        
        

