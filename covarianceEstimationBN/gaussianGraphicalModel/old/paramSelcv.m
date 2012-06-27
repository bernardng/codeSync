
% Lambda Selection
% Input:    X = nxd feature matrix, if X = cell(2,1), X{1} = Xtrain, X{2} = Xtest
%           optMethod = 1:two-metric projection, 2:(Friedman,2007)
%           lambda = 'linear' or user-specified range
%           nLevels = #levels for choosing lambda 
%           nGridPts = #grid points per level
%           modelEvid = function handle of model evidence
%           varargin = variables required for computing user-defined model evidence
% Output:   lambdaBest = optimal lambda
function [lambdaBest,evidBest,Kbest] = paramSel(X,lambdaMax,nLevels,nGridPts,modelEvid,varargin)
if length(X) == 2 % If using cross validation
    Strain = cov(X{1});
    Stest = cov(X{2});
else % If using model evidence
    Strain = cov(X);
end
d = size(Strain,1);
lambdaMin = lambdaMax/100;
evidBest = -inf;
lambdaAcc = [];
for i = 1:nLevels % Number of refinements
    if i == 1
        lambdaGrid = logspace(log10(lambdaMin),log10(lambdaMax),nGridPts);
    else
        if lambdaBest == lambdaMax
            lambdaGrid = logspace(log10(lambdaGrid(2)),log10(lambdaMax),nGridPts+2);
        elseif lambdaBest == lambdaGrid(end)
            lambdaGrid = logspace(log10(lambdaGrid(end)/10),log10(lambdaGrid(end-1)),nGridPts+2);
        else
            ind = find(lambdaGrid == lambdaBest);
            lambdaGrid = logspace(log10(lambdaGrid(ind+1)),log10(lambdaGrid(ind-1)),nGridPts+2);
        end
    end
    lambdaGrid = fliplr(lambdaGrid);
    for j = 1:nGridPts % Number of grid points
        lambda = lambdaGrid(j);
        %             lambda = lambdaMid-(j-round(nGridPts/2))*lambdaStep;
        if i > 1
            if lambda > lambdaMax || sum(lambda == lambdaAcc)
                nSkip = nSkip+1;
                continue; % To avoid computing with the same lambda
            end
        end
        lambdaAcc = [lambdaAcc;lambda]; % Store computed lambdas
        if optMethod == 1 % Graphical LASSO via two-metric projection
            nonZero = find(ones(d));
            funObj = @(x)sparsePrecisionObjBN(x,d,nonZero,Strain);
            if i == 1 && j == 1
                %                     K = inv(oas(Strain)); % Using OAS to initialize
                K = eye(size(Strain));
            elseif i > 1 && j == nSkip+1
                %                     K = Kbest; % Warm start with current best K
                K = eye(size(Strain));
            end
            options.order = -1; % LBFGS
            reg = lambda*~eye(d); % Penalize only off diagonal elements
            K(nonZero) = L1GeneralProjection(funObj,K(nonZero),reg(:),options); % Initialize with previous K to enable warm start
            %                 K(nonZero) = L1GeneralProjectionBN(funObj,K(nonZero),Strain,reg(:),options); % Initialize with previous K to enable warm start
        elseif optMethod == 2 % Friedman's Graphical LASSO
            K = L1precisionBCD(Strain,lambda);
        end
        % Compute model evidence
        if nargin < 7
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
    nSkip = 0; % Count #times lambda is skipped due to >lambdaMax or =previous lambda
end
