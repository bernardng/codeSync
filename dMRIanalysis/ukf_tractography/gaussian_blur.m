% Add a Gaussian blur to the fiber end point
% Input:    i,j,k = coordinate of endpoint
%           template = 3D parcel lable matrix
% Output:   labels = Parcels within the 26-neighborhood
%           weights = weights assigned parcels based on distance
function [labels,weights] = gaussian_blur(i,j,k,template)
nNeigh = 26;
% 26 connected neighbors
neighbor = [-1,1,1; 0,1,1; 1,1,1; -1,0,1; 0,0,1; 1,0,1; -1,-1,1; 0,-1,1; 1,-1,1; ...
            -1,1,0; 0,1,0; 1,1,0; -1,0,0; 1,0,0; -1,-1,0; 0,-1,0; 1,-1,0; -1,1,-1; ...
            0,1,-1; 1,1,-1; -1,0,-1; 0,0,-1; 1,0,-1; -1,-1,-1; 0,-1,-1; 1,-1,-1];
% Weights = exp(-dist) associated with each neighbor
wt = zeros(nNeigh,1);
wt(sum(abs(neighbor),2)==3) = 0.1769;
wt(sum(abs(neighbor),2)==2) = 0.2431;
wt(sum(abs(neighbor),2)==1) = 0.3679;

parcels = zeros(nNeigh,1);
for n = 1:nNeigh
    x = i + neighbor(n,1);
    y = j + neighbor(n,2);
    z = k + neighbor(n,3);
    try
        parcels(n) = template(x,y,z);
    catch ME
        parcels(n) = 0;
    end
end
labels = setdiff(unique(parcels),0);
nLabels = length(labels);
weights = zeros(nLabels,1);
for l = 1:length(labels)
    weights(l) = (parcels==labels(l))'*wt;
end
weights = weights/sum(weights);


