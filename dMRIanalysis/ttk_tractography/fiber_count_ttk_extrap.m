% Count the #fibers going between each pair of parcels in a template
% Input:    nSteps = #steps along tangent to extrapolate, e.g. 2mm resolution, 5 steps = 1cm, 
%           but best to not extrapolate, i.e. set to 0
%           templatePath = filepath to template
%           fiberPath = filepath to fiber
%           connPath = filepath to save fiber count
% Output:   Kfibcnt = fiber count matrix
%           kfiblen = average fiber length between parcels
%           parcelVol = sum of volumes of the connected parcels
% Notes:    Fiber endpoints extrapolated along tangent direction if not on
%           grey matter voxels
function [Kfibcnt,Kfiblen,parcelVol] = fiber_count_ttk_extrap(nSteps,templatePath,fiberPath,connPath)
addpath(genpath('/home/bn228083/code/dMRIanalysis/'));

% Load parcel template
nii = load_untouch_nii(templatePath);
rois = unique(nii.img);
rois(1) = []; % Remove background
nROIs = length(rois);
Kfibcnt = zeros(nROIs,nROIs);
Kfiblen = zeros(nROIs,nROIs);

% Load template in DWI space
nii = load_untouch_nii(templatePath);
template = nii.img;
matrixdim = nii.hdr.dime.dim(2:4);
voxdim = nii.hdr.dime.pixdim(2:4);
[sx,sy,sz] = size(template);
% Compute transform from world space to index space
affineToIndexT = inv([nii.hdr.hist.srow_x;nii.hdr.hist.srow_y;nii.hdr.hist.srow_z;0 0 0 1]);

% Load fiber
fiber = read_fiber(fiberPath,matrixdim,voxdim);
nFiber = length(fiber.fiber);

% Fiber count computation
for n = 1:nFiber
    % This setting is valid for TTK tractography only
    x = -fiber.fiber(n).xyzFiberCoord(:,1);
    y = -fiber.fiber(n).xyzFiberCoord(:,2);
    z = fiber.fiber(n).xyzFiberCoord(:,3);
    ijk = affineToIndexT*[x';y';z';ones(1,length(x))];
%     fiber.fiber(n).xyzFiberCoord = [ijk(2,:)'+1,ijk(1,:)'+1,ijk(3,:)'+1];
    
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
            Kfibcnt(rois==labelStart,rois==labelEnd) = Kfibcnt(rois==labelStart,rois==labelEnd)+1;
            Kfibcnt(rois==labelEnd,rois==labelStart) = Kfibcnt(rois==labelEnd,rois==labelStart)+1;
            Kfiblen(rois==labelStart,rois==labelEnd) = Kfiblen(rois==labelStart,rois==labelEnd)+fiber.fiber(n).nFiberLength;
            Kfiblen(rois==labelEnd,rois==labelStart) = Kfiblen(rois==labelEnd,rois==labelStart)+fiber.fiber(n).nFiberLength;
        end
    end
end

% Compute volume of each pair of parcels
ind = Kfibcnt~=0;
[i,j] = ind2sub(size(Kfibcnt),find(ind));
parcelVol = zeros(nROIs,nROIs);
for k = 1:length(i)
    parcelVol(i(k),j(k)) = sum(template(:)==i(k))+sum(template(:)==j(k));
end

% Compute average fiber length
Kavefiblen = zeros(size(Kfibcnt));
Kavefiblen(ind) = Kfiblen(ind)./Kfibcnt(ind);
Kfiblen = Kavefiblen;

if nargin == 4
    save(connPath,'Kfibcnt','Kfiblen','parcelVol');
end
