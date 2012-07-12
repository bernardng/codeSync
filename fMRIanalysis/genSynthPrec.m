% Generate precision matrix
% Input:    tc = nxd observation matrix, n = #samples, d = #features
%           A = dxd adjacency matrix
%           B = dxd bilateral connection matrix
% Output:   K = (1-gamma(1)-gamma(2))*I - gamma(1) * A - gamma(2) * B
%           Find K that minimizes neg log data likelihood assuming Gaussian
function K = genSynthPrec(tc,A,B)
addpath(genpath('/home/bn228083/matlabToolboxes/markSchmidtCode/'));
[n,d] = size(tc);
S = corr(tc);
f = @(gamma) -log(det(gamma(3)*(eye(d)+gamma(1)*A+gamma(2)*B)))+trace(S*gamma(3)*(eye(d)+gamma(1)*A+gamma(2)*B));
gammaInit = [-0.1;-0.1;0.1];
options.numDiff = 1;
[gamma,evid] = minFunc(f,gammaInit,options);

gammaInit = [0.01;0.01];
proj = @(x) min(max(x,1e-6),1);
[gamma,evid] = minConf_SPG(f,gammaInit,proj,options);

K = eye(d) - gamma(1)*A - gamma(2)*B;


S = corr(tc);
[V,D] = eig(A+B);
f = @(gamma) -d*log(gamma(1))-sum(log(1-gamma(2)*D(eye(d)==1)))+gamma(1)*trace(S*(eye(d)-gamma(2)*(A+B)));
gammaInit = [0.1;0.1];
options.numDiff = 1;
options.progTol = 1e-15;
[gamma,evid] = minFunc(f,gammaInit,options);
K = gamma(1)*(eye(d)-gamma(2)*(A+B));


alpha = -[0.01:0.01:0.1];
gamma = 0.01:0.01:0.1;
nSteps = length(alpha);
[alpha,gamma] = meshgrid(alpha,gamma);
alpha = alpha(:); gamma = gamma(:);
g = zeros(nSteps,1);
for i = 1:length(alpha)
    g(i) = log(det(gamma(i)*(eye(d)-alpha(i)*A-beta(i)*B)))-trace(S*gamma(i)*(eye(d)-alpha(i)*A-beta(i)*B));
end
g(~isreal(g)) = nan;
g = reshape(g,nSteps,nSteps,nSteps);



% For regularizing Laplacian derived from dMRI 
L = diag(sum(Kanat))-Kanat;
[V,D] = eig(L);
f = @(alpha) -d*log(alpha(1)) - sum(log(D(D>0)+alpha(2))) + alpha(1)*trace(S*(L+alpha(2)*eye(d)));
alphaInit = [0.1;0.1];
options.numDiff = 1;
options.maxFunEvals = 5e3;
[alpha,evid] = minFunc(f,alphaInit,options);
K = alpha(1)*(L+alpha(2)*eye(d));