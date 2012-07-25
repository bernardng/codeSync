% Generate precision matrix
% Input:    tc = nxd observation matrix, n = #samples, d = #features
%           A = dxd adjacency matrix
%           B = dxd bilateral connection matrix
% Output:   K = gamma(1)(I - gamma(2) * (A + B))
%           Find K that minimizes neg log data likelihood assuming Gaussian
function K = genSynthPrec(tc,A,B)
addpath(genpath('/home/bn228083/matlabToolboxes/markSchmidtCode/'));
[n,d] = size(tc);
S = corr(tc);
[V,D] = eig(A+B);
f = @(gamma) -d*log(gamma(1))-sum(log(1-gamma(2)*D(eye(d)==1)))+gamma(1)*trace(S*(eye(d)-gamma(2)*(A+B)));
gammaInit = [0.1;0.1];
options.numDiff = 1;
options.progTol = 1e-15;
[gamma,evid] = minFunc(f,gammaInit,options);
K = gamma(1)*(eye(d)-gamma(2)*(A+B));

% S = corr(tc);
% f = @(gamma) -log(det(gamma(3)*(eye(d)+gamma(1)*A+gamma(2)*B)))+trace(S*gamma(3)*(eye(d)+gamma(1)*A+gamma(2)*B));
% gammaInit = [-0.1;-0.1;0.1];
% options.numDiff = 1;
% [gamma,evid] = minFunc(f,gammaInit,options);
% 
% gammaInit = [0.01;0.01];
% proj = @(x) min(max(x,1e-6),1);
% [gamma,evid] = minConf_SPG(f,gammaInit,proj,options);
% 
% K = eye(d) - gamma(1)*A - gamma(2)*B;



