function d = klDiv(A,B)
N = size(A,1);
d = 0.5*(trace(inv(A)*B)-log(det(A))+log(det(B))-N);
% d = 0.5*(trace(inv(A)*B)-trace(log(A))+trace(log(B))-N);