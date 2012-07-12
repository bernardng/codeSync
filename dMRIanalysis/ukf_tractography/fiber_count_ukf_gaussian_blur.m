% Count the #fibers going between each pair of parcels in a template
% Input:    templatePath = filepath to template
%           gmPath = filepath to grey matter mask
%           wmPath = filepath to white matter mask
%           csfPath = filepath to CSF mask
%           fiberPath = filepath to fiber
%           connPath = filepath to save fiber count
% Output:   Kanat = fiber count matrix
% Notes:    A Gaussian is used to account for fiber endpoints uncertainty
function Kanat = fiber_count_ukf_gaussian_blur(templatePath,fiberPath,connPath)
addpath(genpath('/home/bn228083/code/dMRIanalysis/'));

% Load parcel template
nii = load_nii(templatePath);
rois = unique(nii.img);
rois(1) = []; % Remove background
nROIs = length(rois);
Kanat = zeros(nROIs,nROIs);

% Load template in DWI space
nii = load_untouch_nii(templatePath);
template = nii.img;
matrixdim = nii.hdr.dime.dim(2:4);
voxdim = nii.hdr.dime.pixdim(2:4);
[sx,sy,sz] = size(template);
% Compute transform from world space to index space
% affineToIndexT = inv([nii.hdr.hist.srow_x;nii.hdr.hist.srow_y;nii.hdr.hist.srow_z;0 0 0 1]);

% Load fiber
fiber = read_fiber(fiberPath,matrixdim,voxdim);
nFiber = length(fiber.fiber);

% Fiber count computation
for n = 1:nFiber
    % This setting is valid for UKF tractography only
    i = fiber.fiber(n).xyzFiberCoord(:,3);
    j = fiber.fiber(n).xyzFiberCoord(:,2);
    k = fiber.fiber(n).xyzFiberCoord(:,1);
    ijk = [i';j';k'];
    fiber.fiber(n).xyzFiberCoord = round([ijk(2,:)',ijk(1,:)',ijk(3,:)'])+1;
    
    % Convert indices to numbers 
    I = round(ijk(1,:)) + 1; % +1 to account for MATLAB indexing convention
    J = round(ijk(2,:)) + 1;
    K = round(ijk(3,:)) + 1;
    % Ensure indices do not exceed bounding box
    I = min(max(I,1),sx);
    J = min(max(J,1),sy);
    K = min(max(K,1),sz);
    
    % Update anatomical connection matrix for each fiber
    [labelStart,weightStart] = gaussian_blur(I(1),J(1),K(1),template); % Labels of fiber start point
    [labelEnd,weightEnd] = gaussian_blur(I(end),J(end),K(end),template); % Labels of fiber end point
    for p = 1:length(labelStart)
        for q = 1:length(labelEnd)
            Kanat(rois==labelStart(p),rois==labelEnd(q)) = Kanat(rois==labelStart(p),rois==labelEnd(q))+weightStart(p)*weightEnd(q);
            Kanat(rois==labelEnd(q),rois==labelStart(p)) = Kanat(rois==labelEnd(q),rois==labelStart(p))+weightEnd(q)*weightStart(p);
        end
    end
end

if nargin == 3
    save(connPath,'Kanat');
end