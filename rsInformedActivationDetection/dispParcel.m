% Display detected parcels overlaid on T1
% Input:    ContrastNum = contrast# relative to contrastList.mat,
%           activation of each condition not included
%           slice = slice to display
%           percentileRange = percentile range at which parcel declared significant 
%           percentile = percentile at which to threshold
function roiDet = dispParcel(contrastNum,slice,percentileRange,percentile)
addpath(genpath('D:/research/toolboxes/nifti'));
brainSiz = [53,63,46];
thresh = percentileRange==percentile; % Convert percentile threshold to index relative to sig
len = length(percentileRange);

% Load parcel mask
load('D:/research/data/imagen/group/parcel1000Refined');
roiMask = reshape(roiMask,brainSiz);
rois = unique(roiMask); % Parcel numbers
nROIs = length(rois)-1;

% Overlaid detected parcels on T1
fig3D = figure;
figROC = figure;
mni = load_nii('D:/research/data/imagen/group/MNI152_T1_3mm.nii'); mni = mni.img;
figure(fig3D); imagesc(imrotate(mni(:,:,slice),90,'crop')); colormap('gray');
nMethods = 4;
roiDet = cell(nMethods,1);
for method = 1:nMethods
    if method == 1
        methodName = 'OAS';
        colour = cat(3,ones(size(mni(:,:,slice))),zeros(size(mni(:,:,slice))),zeros(size(mni(:,:,slice)))); % Red
        lineColour = 'r';
    elseif method == 2
        methodName = 'GL';
        colour = cat(3,zeros(size(mni(:,:,slice))),zeros(size(mni(:,:,slice))),ones(size(mni(:,:,slice)))); % Blue
        lineColour = 'b';
    elseif method == 3
        methodName = 'Ridge';
        colour = cat(3,zeros(size(mni(:,:,slice))),ones(size(mni(:,:,slice))),zeros(size(mni(:,:,slice)))); % Green        
        lineColour = 'g';
    elseif method == 4
        methodName = 'OLS';
        colour = cat(3,0.5*ones(size(mni(:,:,slice))),0.5*ones(size(mni(:,:,slice))),ones(size(mni(:,:,slice)))); % Mixed
        lineColour = 'k';
    end
    load(['D:/research/projects/imagen/rsInformedActivationDetection/results/sig',methodName,'coi']);
    sig3D = sigTo3D(sig(contrastNum,:,thresh),rois,roiMask);
    nContrasts = size(sig,1);
    figure(fig3D);
    hold on; h = imshow(colour); hold off; set(h,'AlphaData',0.35*imrotate(sig3D(:,:,slice),90,'crop'));    
    figure(figROC); % Displaying up to 50th percentile
    hold on; plot(percentileRange(1:round(len/2))/100,flipud(squeeze(sum(sum(sig(:,:,round(len/2):len)))))/nROIs/nContrasts*100,lineColour);
    
    if 0 % Determine active AAL ROIs
        roiDet{method} = parcelToAAL(sig(contrastNum,:,thresh),rois,roiMask);
    end
end

