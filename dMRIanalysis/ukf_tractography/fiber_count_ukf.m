% Count the #fibers going between each pair of parcels in a template
% Input:    nSteps = #steps along tangent to extrapolate, e.g. 2mm resolution, 5 steps = 1cm
%           templatePath = filepath to template
%           gmPath = filepath to grey matter mask
%           wmPath = filepath to white matter mask
%           csfPath = filepath to CSF mask
%           fiberPath = filepath to fiber
%           connPath = filepath to save fiber count
% Output:   Kanat = fiber count matrix
% Notes:    Fiber endpoints extrapolated along tangent direction if not on
%           grey matter voxels
function Kanat = fiber_count_ukf(nSteps,templatePath,gmPath,wmPath,csfPath,fiberPath,connPath)
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
affineToIndexT = inv([nii.hdr.hist.srow_x;nii.hdr.hist.srow_y;nii.hdr.hist.srow_z;0 0 0 1]);

% Remove subject specific non-gray matter voxels from template
nii = load_untouch_nii(gmPath); gmMask = nii.img;
nii = load_untouch_nii(wmPath); wmMask = nii.img;
nii = load_untouch_nii(csfPath); csfMask = nii.img;
probTotal = gmMask+wmMask+csfMask;
ind = probTotal>0; % Account for voxels with unspecified tissue type
gmMask(ind) = gmMask(ind)./probTotal(ind);
wmMask(ind) = wmMask(ind)./probTotal(ind);
csfMask(ind) = csfMask(ind)./probTotal(ind);
tissueMask = [gmMask(:),wmMask(:),csfMask(:)];
[~,tissue] = max(tissueMask,[],2);
tissue(probTotal==0) = 0; % To remove voxels with unknown tissue type
template(tissue~=1) = 0;

% Load fiber
fiber = read_fiber(fiberPath,matrixdim,voxdim);
nFiber = length(fiber.fiber);

% Fiber count computation
for n = 1:nFiber
    % This setting is valid for UKF tractography only
    i = fiber.fiber(n).xyzFiberCoord(:,2);
    j = fiber.fiber(n).xyzFiberCoord(:,3);
    k = fiber.fiber(n).xyzFiberCoord(:,1);
    ijk = [i';j';k'];
    
    
    fiber.fiber(n).xyzFiberCoord = round(ijk')+1;
        
    
%         x = -fiber.fiber(n).xyzFiberCoord(:,1);
%         y = -fiber.fiber(n).xyzFiberCoord(:,2);
%         z = fiber.fiber(n).xyzFiberCoord(:,3);
%         ijk = affineToIndexT*[x';y';z';ones(1,length(x))];
%         fiber.fiber(n).xyzFiberCoord = [ijk(2,:)',ijk(1,:)',ijk(3,:)'];


    % Convert indices to numbers 
    I = round(ijk(1,:)) + 1; % +1 to account for MATLAB indexing convention
    J = round(ijk(2,:)) + 1;
    K = round(ijk(3,:)) + 1;
    % Ensure indices do not exceed bounding box
    I = min(max(I,1),sx);
    J = min(max(J,1),sy);
    K = min(max(K,1),sz);
    % Update anatomical connection matrix for each fiber
    labelStart = template(I(1),J(1),K(1)); % Label of fiber start point
    
    % Extrapolate along tangent direction if start point not on gray matter voxel
    if labelStart == 0
        tangent = [ijk(1,1)-ijk(1,2);ijk(2,1)-ijk(2,2);ijk(3,1)-ijk(3,2)]; % tangent direction
        tangent = tangent/norm(tangent); % Convert to unit vector
        for s = 1:nSteps
            % Check positive tangent direction
            i = round(ijk(1,1)+s*tangent(1)) + 1; % +1 to account for MATLAB indexing convention
            j = round(ijk(2,1)+s*tangent(2)) + 1;
            k = round(ijk(3,1)+s*tangent(3)) + 1;
            % Ensure indices do not exceed bounding box
            i = min(max(i,1),sx);
            j = min(max(j,1),sy);
            k = min(max(k,1),sz);
            labelStart = template(i,j,k);
            if labelStart ~= 0
                break;
            end
            % Check negative tangent direction
            i = round(ijk(1,1)-s*tangent(1)) + 1; % +1 to account for MATLAB indexing convention
            j = round(ijk(2,1)-s*tangent(2)) + 1;
            k = round(ijk(3,1)-s*tangent(3)) + 1;
            % Ensure indices do not exceed bounding box
            i = min(max(i,1),sx);
            j = min(max(j,1),sy);
            k = min(max(k,1),sz);
            labelStart = template(i,j,k);
            if labelStart ~= 0
                break;
            end
        end
    end
    if labelStart ~= 0 % Skip label extraction of fiber if start point not in ROI
        labelEnd = template(I(end),J(end),K(end)); % Label of fiber end point
        % Extrapolate along tangent direction if end point not on gray matter voxel
        if labelEnd == 0
            tangent = [ijk(1,end)-ijk(1,end-1);ijk(2,end)-ijk(2,end-1);ijk(3,end)-ijk(3,end-1)];
            tangent = tangent/norm(tangent);
            for s = 1:nSteps
                % Check positive tangent direction
                i = round(ijk(1,end)+s*tangent(1)) + 1; % +1 to account for MATLAB indexing convention
                j = round(ijk(2,end)+s*tangent(2)) + 1;
                k = round(ijk(3,end)+s*tangent(3)) + 1;
                % Ensure indices do not exceed bounding box
                i = min(max(i,1),sx);
                j = min(max(j,1),sy);
                k = min(max(k,1),sz);
                labelEnd = template(i,j,k);
                if labelEnd ~= 0
                    break;
                end
                % Check negative tangent direction
                i = round(ijk(1,end)-s*tangent(1)) + 1; % +1 to account for MATLAB indexing convention
                j = round(ijk(2,end)-s*tangent(2)) + 1;
                k = round(ijk(3,end)-s*tangent(3)) + 1;
                % Ensure indices do not exceed bounding box
                i = min(max(i,1),sx);
                j = min(max(j,1),sy);
                k = min(max(k,1),sz);
                labelEnd = template(i,j,k);
                if labelEnd ~= 0
                    break;
                end
            end
        end
        if labelEnd ~= 0
            Kanat(rois==labelStart,rois==labelEnd) = Kanat(rois==labelStart,rois==labelEnd)+1;
            Kanat(rois==labelEnd,rois==labelStart) = Kanat(rois==labelEnd,rois==labelStart)+1;
        end
    end
end

% Plotting volume
slice = round(matrixdim(3)/2); % Display center axial slice
figure;
surf([1 matrixdim(1)],[1 matrixdim(2)],repmat(slice,[2 2]),uint8(template(:,:,slice)),'facecolor','texture');
hold on; 
display_fiber(fiber,1:20:nFiber,'default',0,0,'-');



if nargin == 7
    save(connPath,'Kanat');
end