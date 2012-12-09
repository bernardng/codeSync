% Sparse Group Gaussian Graphical Model with ADMM Consensus Optimatization
% Input:    X = nxdxs matrix, n = #samples, d = #features, s = #subjects
%           lambda = sparsity level
% Output:   Kgrp = dxd sparse group precision matrix
%           converged = flag indicating convergence
function [Kgrp,converged] = sgggmADMM(X,lambda,maxIter)
% Parameters
[n,d,s] = size(X);
rho = 1;
ABSTOL = 1e-4;
RELTOL = 1e-2;
if nargin < 3
    maxIter = 100;
end
% Initialization
K = zeros(d,d,s);
S = zeros(d,d,s);
for sub = 1:s
    [Coas,S(:,:,sub)] = oas(X(:,:,sub));
    K(:,:,sub) = inv(Coas);
end
Kgrp = mean(K,3);
U = zeros(d,d,s);
converged = 0;
for k = 1:maxIter
    disp(['Iteration ',int2str(k)]);
    % K update
    parfor sub = 1:s
        [V,D] = eig(rho*(Kgrp-U(:,:,sub))-S(:,:,sub));
        K(:,:,sub) = V*diag((diag(D)+sqrt(diag(D).^2+4*rho))/(2*rho))*V';
    end
    % Kgrp update
    KgrpOld = Kgrp;
    a = mean(K,3)+mean(U,3)/rho;
    thresh = lambda/(s*rho);
    Kgrp = max(a-thresh,0)-max(-a-thresh,0);
    Kgrp(eye(d)==1) = a(eye(d)==1); % Do not penalize diagonal
    % U update
    for sub = 1:s
        U(:,:,sub) = U(:,:,sub) + rho*(K(:,:,sub)-Kgrp);
    end
    % Residual computation
    rNorm  = norm(mean(K-repmat(Kgrp,[1,1,s]),3),'fro');
    sNorm  = norm(-rho*(Kgrp-KgrpOld),'fro');
    epsPri = sqrt(d*d)*ABSTOL+RELTOL*max(norm(mean(K,3),'fro'),norm(Kgrp,'fro'));
    epsDual = sqrt(d*d)*ABSTOL+RELTOL*norm(rho*mean(U,3),'fro');
    % Convergence check
    if (rNorm < epsPri) && (sNorm < epsDual)
        converged = 1; 
        break;
    end
end

