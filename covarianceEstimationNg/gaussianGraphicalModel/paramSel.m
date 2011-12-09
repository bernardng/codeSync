% Lambda Selection
% Input:    X = nxd feature matrix, if X = cell(2,1), X{1} = Xtrain, X{2} = Xtest
%           optMethod = 1:two-metric projection, 2:(Friedman,2007)
%           lambda = 'linear' or user-specified range
%           nLevels = #levels for choosing lambda 
%           nGridPts = #grid points per level
%           modelEvid = function handle of model evidence
%           varargin = variables required for computing user-defined model evidence
% Output:   lambdaBest = optimal lambda
function [lambdaBest,evidBest,Kbest] = paramSel(X,optMethod,lambda,nLevels,nGridPts,modelEvid,varargin)
if length(X) == 2 % If using cross validation
    Strain = cov(X{1});
    Stest = cov(X{2});
else % If using model evidence
    Strain = cov(X);
end
d = size(Strain,1);
% User-specified lambda
if isnumeric(lambda) 
    evidBest = -inf;
    for i = 1:length(lambda)
        if optMethod == 1
            nonZero = find(ones(d));
            funObj = @(x)sparsePrecisionObj(x,d,nonZero,Strain);
            if i == 1
                K = inv(oas(Strain));
            end
            options.order = -1; % LBFGS
            reg = lambda(i)*~eye(d); % Penalize only off diagonal elements
            K(nonZero) = L1GeneralProjection(funObj,K(nonZero),reg(:),options); % Initialize with previous K to enable warm start
        elseif optMethod == 2
            K = L1precisionBCD(Strain,lambda(i));
        end
        % Compute model evidence
        if nargin < 6
            evid = logDataLikelihood(Stest,K);
        else
            [dummy,evid] = modelEvid(K,varargin{:});
        end
        % Store currently best lambda and precision
        if evidBest < evid 
            evidBest = evid;
            lambdaBest = lambda;
            Kbest = K;
        end
    end
else % Linear scale
    lambdaMax = max(Strain(~eye(d))); % Maximum of the off diagonal
    lambdaMin = lambdaMax/100;
    lambdaMid = (lambdaMax+lambdaMin)/2;
    evidBest = -inf;
    for i = 1:nLevels % Number of refinements
        lambdaStep = (lambdaMax-lambdaMin)/nGridPts/i;
        for j = 1:nGridPts % Number of grid points
            lambda = lambdaMid+(j-round(nGridPts/2))*lambdaStep;
            if i > 1
                if lambda > lambdaMax || lambda == lambdaBest
                    nSkip = nSkip+1;
                    continue; % To avoid computing with the same lambda
                end
            end
            if optMethod == 1 % Graphical LASSO via two-metric projection
                nonZero = find(ones(d));
                funObj = @(x)sparsePrecisionObj(x,d,nonZero,Strain);
                if i == 1 && j == 1
                    K = inv(oas(Strain)); % Using OAS to initialize
                elseif i > 1 && j == nSkip+1
                    K = Kbest; % Warm start with current best K
                end
                options.order = -1; % LBFGS
                reg = lambda*~eye(d); % Penalize only off diagonal elements
                K(nonZero) = L1GeneralProjection(funObj,K(nonZero),reg(:),options); % Initialize with previous K to enable warm start
            elseif optMethod == 2 % Friedman's Graphical LASSO
                K = L1precisionBCD(Strain,lambda);
            end
            % Compute model evidence
            if nargin < 6
                evid = logDataLikelihood(Stest,K);
            else
                [dummy,evid] = modelEvid(K,varargin{:});
            end
            % Store currently best lambda and precision
            if evidBest < evid
                evidBest = evid;
                lambdaBest = lambda;
                Kbest = K;
            end
        end
        lambdaMid = lambdaBest; % Center refinement around best lambda at previous level
        nSkip = 0; % Count #times lambda is skipped due to >lambdaMax or =previous lambda
    end
end