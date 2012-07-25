% Learn anatomical connectivity prior that maximizes data likelihood of RS-fMRI data
% Input:    tc = nxd observation matrix, n = #samples, d = #features
%           K = anatomical connectivity matrix
% Output:   K = alpha(1)(L + alpha(2) * I)
%           Find K that minimizes neg log data likelihood assuming Gaussian
function K = regularizeAnatConn(tc,K)
addpath(genpath('/home/bn228083/matlabToolboxes/markSchmidtCode/'));
[n,d] = size(tc);
S = corr(tc);
L = diag(sum(K))-K;
[V,D] = eig(L);
f = @(alpha) -d*log(alpha(1)) - sum(log(D(D>0)+alpha(2))) + alpha(1)*trace(S*(L+alpha(2)*eye(d)));
alphaInit = [0.1;0.1];
options.numDiff = 1;
options.maxFunEvals = 5e3;
[alpha,evid] = minFunc(f,alphaInit,options);
K = alpha(1)*(L+alpha(2)*eye(d));
