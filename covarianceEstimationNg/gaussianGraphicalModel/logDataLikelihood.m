% Log data likelihood of Gaussian Graphical Model
% Input:    S = empirical covariance of test data
%           K = inverse covariance of training data
% Output:   evid = log data likelihood
function evid = logDataLikelihood(S,K)
% ln(det(.)) = tr(logm(.))
evid = trace(logm(K))-trace(K*S);