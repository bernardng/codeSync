% Calculate the t-value of each voxel
function [beta,tval] = calTval(tc,regressor)

nTpts = size(tc,1);
beta = pinv(regressor)*tc;
S = sum((tc-regressor*beta).^2);
temp = inv(regressor'*regressor);
tval = zeros(size(beta));
for j = 1:size(regressor,2)
    sigj = sqrt(S*temp(j,j)/(nTpts-size(regressor,2)-1)); % Interested in testing beta of regressor
    tval(j,:) = beta(j,:)./sigj;
end