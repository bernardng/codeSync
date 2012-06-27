% Compute dual gap of Gaussian graphical model
% Input:    K = sparse inverse covariance
%           S = empirical covariance
%           lambda = level of sparse regularization
% Output:   dualGap
function dualGap = dualGap(K,S,lambda)
dualGap = abs(S(:)'*K(:)-size(K,1)+lambda(:)'*abs(K(:)));