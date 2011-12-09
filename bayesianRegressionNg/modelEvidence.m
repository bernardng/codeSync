% Estimate optimal level of regularization for Bayesian Regression Model
% Input:    X = nxm regressor matrix, n = #samples, m = #regressors
%           Y = nxd data matrix, d = #features
%           K = dxd prior precision
%           model = {1,...7} see below
%           param = user-specified alpha
%           W = dxd prior precision, use only if model = 4, else leave empty
% Output:   alpha = optimal amount of regularization
%           evid = model evidence
function [alpha,evid] = modelEvidence(X,Y,K,model,param,W)
userAlpha = ~isempty(param);
filepath = 'D:/research/';
addpath(genpath([filepath,'toolboxes/markSchmidtCode']));
[n,m] = size(X);
d = size(K,1);
Id = eye(d);
[eigvec,eigval] = eig(full(K));
eigval = eigval(eigval~=0); % Covert diagonal matrix to vector
if model == 1 % inv(V1) = I, inv(V2) = K
    B = eigvec'*Y'*X/(X'*X)*X'*Y*eigvec;
    % Negative log model evidence
    f = @(alpha) -m*d/2*log(alpha)+m/2*sum(log(1+alpha*eigval))-0.5*sum(B(Id==1)./(1+alpha*eigval));
    alphaInit = 1e-9;
    lb = 0; ub = inf;
elseif model == 2 % inv(V1) = K, inv(V2) = I
    B = eigvec'*K*Y'*X/(X'*X)*X'*Y*K*eigvec;
    % Negative log model evidence
    f = @(alpha) -m*d/2*log(alpha)+m/2*sum(log(eigval+alpha))-0.5*sum(B(Id==1)./(eigval+alpha));
    alphaInit = 1e-9;
    lb = 0; ub = inf;
elseif model == 3 % inv(V1) = inv(V2) = K
    B = eigvec'*K*Y'*X/(X'*X)*X'*Y*K*eigvec;
    % Negative log model evidence
    f = @(alpha) -m*d/2*log(alpha)+m/2*sum(log((1+alpha)*eigval))-0.5*sum(B(Id==1)./((1+alpha)*eigval));
    alphaInit = 1e-9;
    lb = 0; ub = inf;
elseif model == 4 % inv(V1) = I, inv(V2) = K + alpha2*I
    B = eigvec'*Y'*X/(X'*X)*X'*Y*eigvec;
    % Negative log model evidence
    if userAlpha
        f = @(alpha) -m*d/2*log(param)+m/2*sum(log(1+param*eigval+param*alpha))-0.5*sum(B(Id==1)./(1+param*eigval+param*alpha));
        alphaInit = 10^(-9);
        lb = -inf; ub = inf;
    else
        f = @(alpha) -m*d/2*log(alpha(1))+m/2*sum(log(1+alpha(1)*eigval+alpha(1)*alpha(2)))-0.5*sum(B(Id==1)./(1+alpha(1)*eigval+alpha(1)*alpha(2)));
        alphaInit = [1;1]*10^(-9);
        lb = [0;-inf]; ub = [inf;inf];
    end
elseif model == 5 % inv(V1) = K + alpha1*I, inv(V2) = I
    B = eigvec'*Y'*X/(X'*X)*X'*Y*eigvec;
    % Negative log model evidence
    if userAlpha
        f = @(alpha) -m*d/2*log(param)+m/2*sum(log(eigval+alpha+param))-0.5*sum(B(Id==1).*(eigval+alpha).^2./(eigval+alpha+param));
        alphaInit = 10^(-9);
        lb = -inf; ub = inf;
    else
        f = @(alpha) -m*d/2*log(alpha(1))+m/2*sum(log(eigval+alpha(2)+alpha(1)))-0.5*sum(B(Id==1).*(eigval+alpha(2)).^2./(eigval+alpha(2)+alpha(1)));
        alphaInit = [1;1]*10^(-9);
        lb = [0;-inf]; ub = [inf;inf];
    end
elseif model == 6 % inv(V1) = K + alpha1*I, inv(V2) = K + alpha2*I
    B = eigvec'*Y'*X/(X'*X)*X'*Y*eigvec;
    % Negative log model evidence
    if userAlpha
        f = @(alpha) -m*d/2*log(param)+m/2*sum(log((1+param)*eigval+alpha(1)+param*alpha(2)))-0.5*sum(B(Id==1).*(eigval+alpha(2)).^2./((1+param)*eigval+alpha(1)+param*alpha(2)));
        alphaInit = [1;1]*10^(-9);
        lb = [-inf;-inf]; ub = [inf;inf];
    else
        f = @(alpha) -m*d/2*log(alpha(1))+m/2*sum(log((1+alpha(1))*eigval+alpha(2)+alpha(1)*alpha(3)))-0.5*sum(B(Id==1).*(eigval+alpha(2)).^2./((1+alpha(1))*eigval+alpha(2)+alpha(1)*alpha(3)));
        alphaInit = [1;1;1]*10^(-9);
        lb = [0;-inf;-inf]; ub = [inf;inf;inf];
    end
elseif model == 7 % inv(V1) = I, inv(V2) = (1-alpha2)*K + alpha2*I
    B = eigvec'*Y'*X/(X'*X)*X'*Y*eigvec;
    % Negative log model evidence
    if userAlpha
        f = @(alpha) -m*d/2*log(param)+m/2*sum(log(1+param*(1-alpha)*eigval+param*alpha))-0.5*sum(B(Id==1)./(1+param*(1-alpha)*eigval+param*alpha));
        alphaInit = 0.5;
        lb = 0; ub = 1;
    else
        f = @(alpha) -m*d/2*log(alpha(1))+m/2*sum(log(1+alpha(1)*(1-alpha(2))*eigval+alpha(1)*alpha(2)))-0.5*sum(B(Id==1)./(1+alpha(1)*(1-alpha(2))*eigval+alpha(1)*alpha(2)));
        alphaInit = [0.5;0.5];
        lb = [0,0]; ub = [inf,1];
    end
elseif model == 8 % inv(V1) = (1-alpha1)*K + alpha1*I, inv(V2) = I
    B = eigvec'*Y'*X/(X'*X)*X'*Y*eigvec;
    % Negative log model evidence
    if userAlpha
        f = @(alpha) -m*d/2*log(param)+m/2*sum(log((1-alpha)*eigval+alpha+param))-0.5*sum(B(Id==1).*((1-alpha)*eigval+alpha).^2./((1-alpha)*eigval+alpha+param));
        alphaInit = 0.5;
        lb = 0; ub = 1;
    else
        f = @(alpha) -m*d/2*log(alpha(1))+m/2*sum(log((1-alpha(2))*eigval+alpha(2)+alpha(1)))-0.5*sum(B(Id==1).*((1-alpha(2))*eigval+alpha(2)).^2./((1-alpha(2))*eigval+alpha(2)+alpha(1)));
        alphaInit = [0.5;0.5];
        lb = [0,0]; ub = [inf,1];
    end
elseif model == 9 % inv(V1) = (1-alpha1)*K + alpha1*I, inv(V2) = (1-alpha2)*K + alpha2*I
    B = eigvec'*Y'*X/(X'*X)*X'*Y*eigvec;
    % Negative log model evidence
    if userAlpha
        f = @(alpha) -m*d/2*log(param)+m/2*sum(log((1-alpha(1)+param-param*alpha(2))*eigval+alpha(1)+param*alpha(2)))-0.5*sum(B(Id==1).*((1-alpha(1))*eigval+alpha(1)).^2./((1-alpha(1)+param-param*alpha(2))*eigval+alpha(1)+param*alpha(2)));
        alphaInit = [1;1]*10^(-9);
        lb = [0,0]; ub = [1,1];
    else
        f = @(alpha) -m*d/2*log(alpha(1))+m/2*sum(log((1-alpha(2)+alpha(1)-alpha(1)*alpha(3))*eigval+alpha(2)+alpha(1)*alpha(3)))-0.5*sum(B(Id==1).*((1-alpha(2))*eigval+alpha(2)).^2./((1-alpha(2)+alpha(1)-alpha(1)*alpha(3))*eigval+alpha(2)+alpha(1)*alpha(3)));
        alphaInit = [1;1;1]*10^(-9);
        lb = [0,0,0]; ub = [inf,1,1];
    end
else % inv(V1) ~= inv(V2)
    [eigvec,eigval] = jointDiagonalization(full([K,W]));
    B = eigvec'*K*Y'*X/(X'*X)*X'*Y*K*eigvec;
    ind = [Id,Id]==1;
    eigval = eigval(ind); % Extract only the diagonal elements
    eigval1 = eigval(1:end/2);
    eigval2 = eigval(end/2+1:end);
    f = @(alpha) -m*d/2*log(alpha)+m/2*sum(log(eigval1+alpha*eigval2))-0.5*sum(B(Id==1)./(eigval1+alpha*eigval2));
    alphaInit = 1e-9;
end
% Min negative log model evidence
options.numDiff = 1;
[alpha,evid] = minFunc(f,alphaInit,options);
% [alpha,evid] = fmincon(f,alphaInit,[],[],[],[],lb,ub);
if userAlpha
    alpha = [param;alpha];
end
for a = 1:length(alpha)
    disp(['Optimal alpha',num2str(a),' = ',num2str(alpha(a))]);
end
evid = -evid; % Account for sign flip to use minFunc()
disp(['Model Evidence = ',num2str(evid)]);
