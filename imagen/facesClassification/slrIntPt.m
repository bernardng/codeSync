% Sparse Logistic Regression
% Input:    X = mxn feature matrix, m = #samples, n = #features, last feature = a column of ones
%           b = mx1 label vector = {-1,1}
%           lambda = amount of sparse penalization
%           maxIter = maximum number of iterations
%           tol = duality gap 
% Output:   w = nx1 classifier
% Note: Implementation based on Koh et al., "An Interior-Point Method
% for Large-Scale l1-Regularized Logistic Regression," J. Mach. Learn., 2007

function w = slrIntPt(X,b,lambda,maxIter,tol)
netwPath = '/home/bn228083/';
addpath(genpath([netwPath,'MatlabToolboxes/markSchmidtCode']));
alpha = 0.01;
beta = 0.5;
maxk = 200;
mu = 2;
smin = 0.5;
if nargin < 4
    maxIter = 1000;
end
if nargin < 5
    tol = 1e-5;
end
[m,n] = size(X);
A = (b*ones(1,n-1)).*X(:,1:end-1);
lambdaMax = 1/m*max(X(:,1:n-1)'*0.5*b);
lambda = lambda*lambdaMax;

% Initialization
t = 1/lambda;
v = 0; % v = log(m+/m-) = log(1) = 0, since have even #samples for both classes
w = zeros(n-1,1);
u = ones(n-1,1);

for iter = 1:maxIter
    % Compute search direction
    plog = sigmoid(b.*(X*[w;v]));
    g1 = -(t/m)*b'*(ones(m,1)-plog);
    g2 = -(t/m)*A'*(ones(m,1)-plog)+2*w./(u.^2-w.^2);
    g3 = t*lambda*ones(n-1,1)-2*u./(u.^2-w.^2);
    [f,df,d2f] = neglogSigmoid(b.*(X*[w;v]));
    D0 = 1/m*diag(d2f);
    D1 = diag(2*(u.^2+w.^2)./((u.^2-w.^2).^2));
    D2 = diag(-4*u.*w./((u.^2-w.^2).^2));
    D3 = D1-D2*(D1^-1)*D2;
    Sinv = (D3^-1)-(D3^-1)*A'*(((1/t)*(D0^-1)+A*(D3^-1)*A')^-1)*A*(D3^-1);
    dv = ((t*b'*D0*b-t^2*b'*D0*A*Sinv*A'*D0*b)^-1)*(-g1+t*b'*D0*A*Sinv*(g2-D2*(D1^-1)*g3));
    dw = -Sinv*(g2-D2*(D1^-1)*g3+t*A'*D0*b*dv);
    du = -(D1^-1)*(g3+D2*dw);
    % Backtracking line search
    for k = 1:maxk
        if evalPhi(v+beta^k*dv,w+beta^k*dw,u+beta^k*du,X,b,lambda,t) <= ...
            evalPhi(v,w,u,X,b,lambda,t)+alpha*beta^k*[g1;g2;g3]'*[dv;dw;du]
            disp('Amijo condition satisfied');
            break;
        end
    end
    % Update v,w,u
    v = v+beta^k*dv;
    w = w+beta^k*dw;
    u = u+beta^k*du;
    % Compute optimal v
    h = @(vOpt)(mean(log(1+exp(-b.*(X*[w;vOpt])))));
    options.numDiff = 1;
    vOpt = minFunc(h,v,options);
    % Compute dual feasible point theta
    plogvOpt = sigmoid(b.*(X*[w;vOpt]));
    s = min(m*lambda/max(A'*(ones(m,1)-plogvOpt)),1);
    theta = (s/m)*(ones(m,1)-plogvOpt);
    % Evaluate duality gap neta
    neta = mean(neglogSigmoid(b.*(X*[w;v])))+mean(nentropy(-m*theta))+lambda*sum(abs(w))
    if abs(neta) < tol
        disp(['Duality gap below ',num2str(tol)]);
        break;
    else
        if s >= smin
            t = max(mu*min(2*n/neta,t),t);
        end
    end 
end
disp(['Number of iterations ',int2str(iter)]);
thresh = abs(1/m*(A'*(ones(m,1)-sigmoid(b.*(X*[w;v])))))>0.9999*lambda;
w = w.*thresh;
w = [w;v];

function p = sigmoid(a)
p = 1./(1+exp(-a));

function [f,df,d2f] = neglogSigmoid(a)
f = log(1+exp(-a));
df = -1./(1+exp(a));
d2f = 1./(2+exp(a)+exp(-a));

function phi = evalPhi(v,w,u,X,b,lambda,t)
phi = t*mean(neglogSigmoid(b.*(X*[w;v])))+t*lambda*sum(u)-sum(log((u+w).*(u-w)));

function ent = nentropy(y)
ent = inf*ones(size(y));
for i = 1:length(y)
    if (y(i)>-1)&&(y(i)<0)
        ent(i) = -y(i)*log(-y(i))+(1+y(i))*log(1+y(i));
    elseif (y(i)==-1)||(y(i)==0)
        ent(i) = 0;
    end
end

