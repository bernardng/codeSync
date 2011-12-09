% Find AAL labels of detected parcels
% Input:    sig = 1xd binary map, 1 => activated, 0 => non-activated
%           rois = parcel numbers
%           roiMask = 3D parcel mask
% Output:   labels = detected AAL ROIs
function roiDet = parcelToAAL(sig,rois,roiMask)
% Load AAL template
aal = load_nii('D:/research/data/imagen/aalResampled/aalResampled.nii'); aal = aal.img; 
load('D:/research/data/imagen/aalResampled/roiLabels'); % Load AAL labels
load('D:/research/data/imagen/aalResampled/roiLabelNum'); % Load AAL ROI numbers
% Find activated parcel numbers
parcelDet = rois([0,sig]==1); 
% Find corresponding AAL number
aalDet = []; 
for i = 1:length(parcelDet)
    ind = roiMask == parcelDet(i);
    aalDet = [aalDet;aal(ind)]; % In AAL numbering, e.g. 9170 for CER
end
aalDet = unique(aalDet); % Detected parcels in AAL label numbers
roiDet = zeros(length(aalDet),1); % Convert to {1,...,116}
for i = 1:length(aalDet) 
    ind = find(labelNum==aalDet(i));
    if ~isempty(ind)
        roiDet(i) = ind;
    else
        roiDet(i) = 0;
    end
end
roiDet = sort(roiDet);
roiDet(roiDet==0) = []; % Account for how each AAL ROIs might comprise multiple parcels
roiDet = labels(roiDet,2);
