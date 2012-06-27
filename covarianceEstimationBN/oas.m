% Oracle Approximating Shrinkage (OAS)
% Input: tc = nxp, n = #samples, p = #features
% Output: C = pxp, well-conditioned covariance matrix
% Reference: Chen et al., Shrinkage Algorithms for MMSE Covariance Estimation, TSP, 2010
function C = oas(tc)
[n,p] = size(tc);
S = cov(tc); % Empirical covariance
F = trace(S)*eye(p)/p; % Most well-conditioned estimate
trS2 = trace(S*S);
tr2S = trace(S)^2;
rho = min(((1-2/p)*trS2+tr2S)/((n+1-2/p)*(trS2-tr2S/p)),1); % Relative weighting
C = (1-rho)*S+rho*F;
