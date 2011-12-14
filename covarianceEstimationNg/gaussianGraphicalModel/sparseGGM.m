% Sparse Gaussain graphical model via two-metric projection or (Friedman,2007)
% Input:    X = nxd feature matrix, n = #samples, d = #features
%           paramSelMethod = 1:cross validation, 2:model evidence
%           optMethod = 1:two-metric projection, 2: (Friedman,2007)
%           param = 'linear' scale or user-specified lambda range
%           kFolds = #folds for cross validation
%           nLevels = #levels for choosing lambda 
%           nGridPts = #grid points per level
%           modelEvid = function handle of user-defined model evidence
%           varargin = variables required for computing user-defined model evidence
% Output:   K = dxd sparse inverse covariance matrix
function K = sparseGGM(X,paramSelMethod,optMethod,param,kFolds,nLevels,nGridPts,modelEvid,varargin)
localpath = '/volatile/bernardng/';
addpath(genpath([localpath,'matlabToolboxes/markSchmidtCode']));
[n,d] = size(X);
% Determine optimal amount of regularization
if paramSelMethod == 1 % Cross validation 
    [trainInd,testInd] = cvSeq(n,kFolds);
    lambda = zeros(kFolds,1);
    evid = zeros(kFolds,1);
    Xsplit = cell(2,1);
    for fold = 1:kFolds
        Xsplit{1} = X(trainInd{fold},:);
        Xsplit{2} = X(testInd{fold},:);
        [lambda(fold),evid(fold)] = paramSel(Xsplit,optMethod,param,nLevels,nGridPts);
    end
    lambdaBest = evid'*lambda/sum(evid); % Weighted average lambda
    % Compute sparse inverse covariance using optimal lambda
    S = cov(X); % Empirical covariance with all samples
    if optMethod == 1
        nonZero = find(ones(d));
        funObj = @(x)sparsePrecisionObj(x,d,nonZero,S);
        options.order = -1; % LBFGS
        reg = lambdaBest*~eye(d); % Penalize only off diagonal elements
        K = eye(d); % Initialization
        K(nonZero) = L1GeneralProjection(funObj,K(nonZero),reg(:),options);
    else
        K = L1precisionBCD(S,lambdaBest);
    end
elseif paramSelMethod == 2 % Model evidence
    [lambdaBest,evidBest,K] = paramSel(X,optMethod,param,nLevels,nGridPts,modelEvid,varargin{:});
end

    
    
        
        
        
        
        

