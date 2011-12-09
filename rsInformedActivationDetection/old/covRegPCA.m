% Regularizing empirical covariance matrix using PCA
ind = 1:nComp;
[u,s,v] = svd(Y);
resid = Y-u(:,ind)*s(ind,ind)*v(:,ind)';
C = v(:,ind)*s(ind,ind)*s(ind,ind)*v(:,ind)'+diag(sqrt(sum(resid.*resid)));