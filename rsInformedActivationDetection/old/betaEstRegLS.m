% Estimate beta with resting state connectivity as regularization
function [F,G] = betaEstRegLS(beta,X,Y,K,L,alpha)
nConds = size(X,2);
nROIs = size(Y,2);
beta = reshape(beta,nConds,nROIs);

sig = sqrt(K);
F = norm((Y-X*beta)*sig,'fro')+alpha*trace(beta*L*beta');
G = -2*X'*(Y-X*beta)*K+2*alpha*beta*L;
G = G(:);

