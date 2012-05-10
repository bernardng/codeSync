% Sparse Gaussian graphical models solved with 2nd order method
% Input:    S = pxp empirical covarianc
%           lambda = pxp level of sparsity
%           X0 = pxp initial precision matrix
%           tol = convergence based on dual gap
%           maxIter = maximum number of iterations
% Output:   X = pxp sparse inverse covariance
% Note:     Algorithm based on CJ Hsieh, MA Sustik, IS Dhillon, and P
% Ravikumar, "Sparse Inverse Covariance Matrix Estimation Using Quadratic
% Approximation," NIPS 2011.

function X = sggmQUIC(S,lambda,X0,tol,maxIter)

p = size(S,1);
if numel(lambda) == 1
    lambda = lambda*~eye(p);
end
if nargin < 3
    X0 = eye(p);
end
if nargin < 4
    tol = 1e-5;
end
if nargin < 5
    maxIter = 100;
end
lineMaxIter = 20; 

X = X0;
W = X^-1;
beta = 0.5;
objNewPrev = inf;
for k = 1:maxIter
    grad = S-W;
    D = zeros(p); U = zeros(p); normD = 0; 
    % Find Newton direction
    if k == 1 && sum(X0(~eye(p)))==0 % If X0 = diagonal matrix
        % Dij for off diagonal elements
        for i = 1:p 
            for j = 1:i-1
                a = W(i,i)*W(j,j);
                b = S(i,j);
                l = lambda(i,j)/a;
                f = b/a;
                if f < 0
                    mu = -f-l;
                    if mu < 0
%                         mu = 0;
                        D(i,j) = -X(i,j);
                    else
                        D(i,j) = D(i,j)+mu;
                    end
                else
                    mu = -f+l;
                    if mu > 0
%                         mu = 0;
                        D(i,j) = -X(i,j);
                    else
                        D(i,j) = D(i,j)+mu;
                    end
                end
            end
        end
        % Dij for diagonal elements
        for i = 1:p
            a = W(i,i)^2;
            b = S(i,i)-W(i,i);
            l = lambda(i,i)/a;
            c = X(i,i);
            f = b/a;
            if c > f
                mu = -f-l;
                if c+mu < 0
                    D(i,i) = -X(i,i);
                    continue;
                end
            else
                mu = -f+l;
                if c+mu > 01e15;
                    D(i,i) = -X(i,i);
                    continue;
                end
            end
            D(i,i) = D(i,i)+mu;
        end
    else
        ind = find(abs(grad)>(lambda-1e-2)|X~=0);
        nConn = length(ind); % Number of connections in free set
        for ik = 1:1+round(k/3)
            diffD = 0;
            % Divide connections into fixed and free sets
%             ind = ind(randperm(nConn));
            [i,j] = ind2sub([p,p],ind); % free set
            for n = 1:nConn
                % mu = -c+sign(c-b/a)*max(abs(c-b/a)-lambda(i(n),j(n))/a,0)
                if j(n) <= i(n) % To avoid Dij update twice
                    a = W(i(n),j(n))^2;
                    if i(n)~=j(n)
                        a = a+W(i(n),i(n))*W(j(n),j(n));
                    end
                    b = S(i(n),j(n))-W(i(n),j(n))+W(i(n),:)*U(:,j(n));
                    l = lambda(i(n),j(n))/a;
                    c = X(i(n),j(n))+D(i(n),j(n));
                    f = b/a;
                    normD = normD-abs(D(i(n),j(n)));
                    if c > f
                        mu = -f-l;
                        if c+mu < 0
                            mu = -c;
                            D(i(n),j(n)) = -X(i(n),j(n));
                        else
                            D(i(n),j(n)) = D(i(n),j(n))+mu;
                        end
                    else
                        mu = -f+l;
                        if c+mu > 0
                            mu = -c;
                            D(i(n),j(n)) = -X(i(n),j(n));
                        else
                            D(i(n),j(n)) = D(i(n),j(n))+mu;
                        end
                    end
                    diffD = diffD+abs(mu);
                    normD = normD+abs(D(i(n),j(n)));
                    if mu~=0
                        U(i(n),:) = U(i(n),:)+mu*W(j(n),:);
                        if i(n)~=j(n)
                            U(j(n),:) = U(j(n),:)+mu*W(i(n),:);
                        end
                    end
                end
            end
            if diffD < normD*5e-2
%                 disp(['Newton direction converged in ',int2str(ik),' iterations.']);
                break;
            end
        end
    end
    % Symmetrizing D
    D = D+D';
    ind = eye(p)==1;
    D(ind) = D(ind)/2;
    
    % Armijo-rule to select step size
    if k == 1
        objOld = objective(X,S,lambda);
    end
    for ik = 1:lineMaxIter
        [objNew,R] = objective(X+beta^(ik-1)*D,S,lambda);
        if isinf(objNew)
            continue;
        end
        if objNew < objOld+0.001*beta^(ik-1)*(grad(:)'*D(:)+lambda(:)'*abs(X(:)+D(:))-lambda(:)'*abs(X(:)))
%             disp(['Armijo-rule satisfied in ', int2str(ik),' iterations.']);
            objOld = objNew;
            break;
        end
        if objNewPrev < objNew
            break;
        end
        objNewPrev = objNew;
    end
    X = X+beta^(ik-1)*D;
    dualGap = estDualGap(X,S,lambda)
    if dualGap < tol
        break;
    end
    W = (R^-1)*(R^-1)'; % inverse of X
%     disp(['Newton iteration ',int2str(k)]);
end

% Compute objective of sparse Gaussian graphical model
function [obj,R] = objective(X,S,lambda)
p = size(X,1);
try
    R = chol(X);
    obj = -2*sum(log(R(eye(p)==1)))+S(:)'*X(:)+lambda(:)'*abs(X(:));
catch ME
    R = [];
    obj = inf;
end

% Compute dual gap of sparse Gaussian graphical model
function dualGap = estDualGap(X,S,lambda)
dualGap = abs(S(:)'*X(:)-size(X,1)+lambda(:)'*abs(X(:)));
