% Estimate beta with resting state connectivity as regularization
function beta = betaEstReg(X,Y,L,alpha)
nConds = size(X,2);
nVox = size(Y,2);
cvx_begin
    variable beta(nConds,nVox);
    for cond = 1:nConds
        bLbT(cond) = beta(cond,:)*L*beta(cond,:)';
    end
    minimize( norm((Y-X*beta),'fro')+alpha*sum(bLbT) );   
cvx_end


