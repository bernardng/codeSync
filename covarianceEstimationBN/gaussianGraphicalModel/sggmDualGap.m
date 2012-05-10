% Compute dual gap of Graphical LASSO
% Input:    K = sparse inverse covariance
%           S = empirical covariance
%           lambda = level of sparse regularization
% Output:   dualGap
function dualGap = estDualGap(K,S,lambda)
dualGap = abs(trace(S*K)-size(K,1)+sum(lambda(:).*abs(K(:))));