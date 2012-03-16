% Log data likelihood of Gaussian Graphical Model
% Input:    S = empirical covariance of test data
%           K = inverse covariance of training data
% Output:   evid = log data likelihood
function evid = logDataLikelihood(S,K)
try
    L = chol(K);
    logdetK = 2*sum(log(diag(L)));
    evid = logdetK-trace(S*K);
catch ME
    evid = -inf;
end
    
% % Adopted from numpy
% logdetK = sum(log(K(eye(size(K))==1)));
% a = exp(logdetK/size(K,1));
% d = det(K/a);
% logdetK = logdetK + log(d);
% if isfinite(logdetK)
%     evid = logdetK-sum(sum(S.*K));
% else
%     evid = -inf;
% end
%     
    
    
    
