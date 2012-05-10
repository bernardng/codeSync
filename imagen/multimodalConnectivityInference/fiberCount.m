% Counting #fibers going through each pair of ROIs
clear all; %close all;
localpath = '/volatile/bernardng/';
netwpath = '/home/bernard/';
addpath(genpath([netwpath,'matlabToolboxes/dwiUtils']));
addpath(genpath([netwpath,'matlabToolboxes/nifti']));
fid = fopen([localpath,'data/imagen/subjectLists/subjectListDWI.txt']);
nSubs = 60;
sublist = cell(nSubs,1);
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end

subs = 1:60; 

nSteps = 5; % 1 step = 2mm
% nii = load_nii([localpath,'templates/freesurfer/cort_subcort_333.nii']);
nii = load_nii([localpath,'data/imagen/group/parcel500refined.nii']);
rois = unique(nii.img);
rois(1) = []; % Remove background
nROIs = length(rois);
D = zeros(nROIs,nROIs,nSubs);
for sub = subs
    % Load template in DWI space
%     nii = load_untouch_nii([localpath,'data/imagen/',sublist{sub},'/dwi/affFStemplate.nii']);
    nii = load_untouch_nii([localpath,'data/imagen/',sublist{sub},'/dwi/affparcel500.nii']);
    template = nii.img;
    matrixdim = nii.hdr.dime.dim(2:4);
    voxdim = nii.hdr.dime.pixdim(2:4);
    [sx,sy,sz] = size(template);
    % Compute transform from world space to index space
    affineToIndexT = inv([nii.hdr.hist.srow_x;nii.hdr.hist.srow_y;nii.hdr.hist.srow_z;0 0 0 1]);
    
    % Remove subject specific non-gray matter voxels from template
    nii = load_untouch_nii([localpath,'data/imagen/',sublist{sub},'/dwi/affgmMask.nii']);
    gmMask = nii.img;
    nii = load_untouch_nii([localpath,'data/imagen/',sublist{sub},'/dwi/affwmMask.nii']);
    wmMask = nii.img;
    nii = load_untouch_nii([localpath,'data/imagen/',sublist{sub},'/dwi/affcsfMask.nii']);
    csfMask = nii.img;
    probTotal = gmMask+wmMask+csfMask;
    ind = probTotal>0; % Account for voxels with unspecified tissue type
    gmMask(ind) = gmMask(ind)./probTotal(ind);
    wmMask(ind) = wmMask(ind)./probTotal(ind);
    csfMask(ind) = csfMask(ind)./probTotal(ind);
    tissueMask = [gmMask(:),wmMask(:),csfMask(:)];
    [dummy,tissue] = max(tissueMask,[],2);
    tissue(probTotal==0) = 0; % To remove voxels with unknown tissue type
    template(tissue~=1) = 0;
    
    % Load fiber
%     fiber = readFiber([localpath,'data/imagen/',sublist{sub},'/dwi/fibersDense.fib'],matrixdim,voxdim);
%     load([localpath,'data/imagen/',sublist{sub},'/dwi/fibersDense.mat']);
    load([localpath,'data/imagen/',sublist{sub},'/dwi/groupFibers.mat']);
    nFiber = length(fiber.fiber);
    % Transform fiber world space coordinates to indices
    for n = 1:nFiber
        x = -fiber.fiber(n).xyzFiberCoord(:,1);
        y = -fiber.fiber(n).xyzFiberCoord(:,2);
        z = fiber.fiber(n).xyzFiberCoord(:,3);
        ijk = affineToIndexT*[x';y';z';ones(1,length(x))];
        % Convert ijk to integer
        I = round(ijk(1,:)) + 1; % +1 to account for MATLAB convention
        J = round(ijk(2,:)) + 1;
        K = round(ijk(3,:)) + 1;
        I = min(max(I,1),sx); 
        J = min(max(J,1),sy);
        K = min(max(K,1),sz);
        % Update anatomical connection matrix for each fiber
        labelStart = template(I(1),J(1),K(1)); % Label of fiber start point
        % Extrapolate along tangent direction
        if labelStart == 0
            tangent = [ijk(1,1)-ijk(1,2);ijk(2,1)-ijk(2,2);ijk(3,1)-ijk(3,2)]; % tangent direction
            tangent = tangent/norm(tangent); % Convert to unit vector
            for s = 1:nSteps
                % Check positive tangent direction
                i = round(ijk(1,1)+s*tangent(1)) + 1; % +1 to account for MATLAB convention
                j = round(ijk(2,1)+s*tangent(2)) + 1;
                k = round(ijk(3,1)+s*tangent(3)) + 1;
                % Ensure extrapolated start point is within template
                i = min(max(i,1),sx);
                j = min(max(j,1),sy);
                k = min(max(k,1),sz);
                labelStart = template(i,j,k);
                if labelStart ~= 0
                    break;
                end
                % Check negative tangent direction
                i = round(ijk(1,1)-s*tangent(1)) + 1; % +1 to account for MATLAB convention
                j = round(ijk(2,1)-s*tangent(2)) + 1;
                k = round(ijk(3,1)-s*tangent(3)) + 1;
                % Ensure extrapolate start point is within template
                i = min(max(i,1),sx);
                j = min(max(j,1),sy);
                k = min(max(k,1),sz);
                labelStart = template(i,j,k);
                if labelStart ~= 0
                    break;
                end
            end
        end
        if labelStart ~= 0 % Skip label extraction of fiber end pountitledint if start point not in ROI
            labelEnd = template(I(end),J(end),K(end)); % Label of fiber end point
            % Extrapolate along tangent direction
            if labelEnd == 0
                tangent = [ijk(1,end)-ijk(1,end-1);ijk(2,end)-ijk(2,end-1);ijk(3,end)-ijk(3,end-1)];
                tangent = tangent/norm(tangent);
                for s = 1:nSteps
                    % Check positive tangent direction
                    i = round(ijk(1,end)+s*tangent(1)) + 1; % +1 to account for MATLAB convention
                    j = round(ijk(2,end)+s*tangent(2)) + 1;
                    k = round(ijk(3,end)+s*tangent(3)) + 1;
                    % Ensure extrapolate end point is within template
                    i = min(max(i,1),sx);
                    j = min(max(j,1),sy);
                    k = min(max(k,1),sz);
                    labelEnd = template(i,j,k);
                    if labelEnd ~= 0
                        break;
                    end
                    % Check negative tangent direction
                    i = round(ijk(1,end)-s*tangent(1)) + 1; % +1 to account for MATLAB convention
                    j = round(ijk(2,end)-s*tangent(2)) + 1;
                    k = round(ijk(3,end)-s*tangent(3)) + 1;
                    % Ensure extrapolate end point is within template
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
                D(rois==labelStart,rois==labelEnd,sub) = D(rois==labelStart,rois==labelEnd,sub)+1;
                D(rois==labelEnd,rois==labelStart,sub) = D(rois==labelEnd,rois==labelStart,sub)+1;
            end
        end
    end
    Kanat = D(:,:,sub);
%     save([localpath,'data/imagen/',sublist{sub},'/dwi/K_anat_parcel500.mat'],'Kanat');
%     save([localpath,'data/imagen/',sublist{sub},'/dwi/K_anatDense_roi_fs.mat'],'Kanat');
    save([localpath,'data/imagen/',sublist{sub},'/dwi/K_anatGroup_parcel500.mat'],'Kanat');
%     figure; imagesc(Kanat~=0);
    % For testing
%     load([localpath,'data/imagen/',sublist{sub},'/precMat/K_rest_roi_fs_quic355_cv.mat']);
%     load([localpath,'data/imagen/',sublist{sub},'/restfMRI/K_rest_roi_fs_GV355_cv.mat']);
%     KrestAcc(:,:,sub) = Krest;
    disp(['Done subject',int2str(sub)]);
end




