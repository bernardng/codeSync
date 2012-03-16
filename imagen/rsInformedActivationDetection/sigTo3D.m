% Convert binary vector "sig" to 3D for visualization
% Input:    sig = 1xd binary map, 1 => activated, 0 => non-activated
%           rois = parcel numbers
%           roiMask = 3D parcel mask
% Output:   sig3D = sig in 3D
function sig3D = sigTo3D(sig,rois,roiMask)
roiDet = rois([0,sig]==1); % Find activated parcel#, 0 accounts for background
sig3D = zeros(size(roiMask)); % 0/1 3D map
for i = 1:length(roiDet)
    ind = roiMask == roiDet(i); % Find 3D location of each detected parcel
    sig3D = sig3D|reshape(ind,size(roiMask)); % Combine all detected parcels into a single map
end