% Bayesian Regression
% Input:    X = nxm regressor matrix, n = #samples, m = #regressors
%           Y = nxd data matrix, d = #features
%           K = dxd prior precision
% Output:   beta = dxm posterior regression coefficients
function beta = bayesianRegression(X,Y,K)
d = size(K,1);
alpha = modelEvidence(X,Y,K);
% inv(V1) = I, inv(V2) = K
V1inv = eye(d);
V2inv = K;
beta = (V1inv+alpha*V2inv)\(V1inv*Y'*X)/(X'*X);
