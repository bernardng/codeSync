% Estimate optimal level of regularization for Bayesian Regression Model
% Input:    X = nxm regressor matrix, n = #samples, m = #regressors
%           Y = nxd data matrix, d = #features
%           K = dxd prior precision
% Output:   alpha = optimal amount of regularization
%           evid = model evidence
function [alpha,evid] = modelEvidence(X,Y,K)
filepath = '/home/bn228083/';
addpath(genpath([filepath,'matlabToolboxes/markSchmidtCode']));

[n,m] = size(X);
d = size(K,1);
[eigvec,eigval] = eig(full(K));
eigval = eigval(eigval~=0); % Covert diagonal matrix to vector
B = eigvec'*Y'*X/(X'*X)*X'*Y*eigvec;
% Negative log model evidence
f = @(alpha) -m*d/2*log(alpha)+m/2*sum(log(1+alpha*eigval))-0.5*sum(B(eye(d)==1)./(1+alpha*eigval));
alphaInit = 0.01;

% Min negative log model evidence
options.numDiff = 1;
[alpha,evid] = minFunc(f,alphaInit,options);

% alpha = 0.001;
% proj = @(x) min(max(x,1e-6),1e-1);
% [alpha,evid] = minConf_SPG(f,alphaInit,proj,options);

% Output optimal alpha and associating model evidence
disp(['Optimal alpha = ',num2str(alpha)]);
evid = -evid; % Account for sign flip to use minFunc()
disp(['Model Evidence = ',num2str(evid)]);
