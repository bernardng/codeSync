% Testing Projected Quasi-Newton Method for Sparse Inverse Covariance Estimation
addpath(genpath('D:\research\toolboxes\projectedQuasiNewton\'));
fid = fopen('D:\research\restStateTaskInteg\imagenData\code\subjectList.txt');
nSubs = 65;
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end
nParcel = 1000;
sub = 1;
load(strcat('D:/research/restStateTaskInteg/imagenData/',sublist{sub},'/tcRestParcel',int2str(nParcel)));
S = cov(tcRest);
funObj = @(K)logdetFunction(K,S);
lambda = 10;
nROIs = size(tcRest,2);
nBlockElements = ones(nROIs,1);
lambdaBlock = setdiag(lambda*nBlockElements*nBlockElements',lambda);
lambdaBlock = lambdaBlock(:);
W0 = lambda*eye(nROIs);
% funProj = @(K)projectLinf2(K,nROIs,nBlockElements,lambdaBlock);
% W(:) = minConF_PQN(funObj,W0(:),funProj);
lambdaFull = lambda*ones(nROIs);
funProj = @(K)boundProject(K,-lambdaFull(:),lambdaFull(:));
W = minConF_SPG(funObj,W0(:),funProj);
Kspg = inv(S+reshape(W,[nROIs nROIs]));



