% Counting #fibers going through each pair of ROIs
clear all; close all;
filepath = '/volatile/bernardng/';
addpath(genpath([filepath,'matlabToolboxes/dwiUtils']));
addpath(genpath([filepath,'matlabToolboxes/nifti']));
fid = fopen([filepath,'data/imagen/subjectLists/subjectListDWI.txt']);
nSubs = 60;
sublist = cell(nSubs,1);
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end

subs = [56:60];

for sub = subs
    % Load template in DWI space
%     nii = load_untouch_nii([filepath,'data/imagen/',sublist{sub},'/dwi/affparcel500.nii']);
    nii = load_untouch_nii([filepath,'data/imagen/',sublist{sub},'/dwi/wmMask_rs_aff_bin.nii']);
    template = nii.img;
    matrixdim = nii.hdr.dime.dim(2:4);
    voxdim = nii.hdr.dime.pixdim(2:4);
    % Compute transform from world space to index space
    affineToIndexT = inv([nii.hdr.hist.srow_x;nii.hdr.hist.srow_y;nii.hdr.hist.srow_z;0 0 0 1]);
    % Plotting template
    slice = round(matrixdim(3)/2);
    figure;
    surf([1 matrixdim(1)],[1 matrixdim(2)],repmat(slice,[2 2]),uint8(template(:,:,slice)),'facecolor','texture');
    
    % Load fiber
%     fiber = readFiber([filepath,'data/imagen/',sublist{sub},'/dwi/groupFibers.fib'],matrixdim,voxdim);
%     fiber = readFiber([filepath,'data/imagen/',sublist{sub},'/dwi/fibersDense.fib'],matrixdim,voxdim);
    fiber = readFiber([filepath,'data/imagen/',sublist{sub},'/dwi/results_ukf/tracks_ukf.fib'],matrixdim,voxdim);
%     save([filepath,'data/imagen/',sublist{sub},'/dwi/groupFibers.mat'],'fiber')
%     load([filepath,'data/imagen/',sublist{sub},'/dwi/fibersDense.mat']);
%     load([filepath,'data/imagen/',sublist{sub},'/dwi/groupFibers.mat']);
    nFiber = length(fiber.fiber);

    XYZ = [];
    % Transform fiber world space coordinates to indices
    for n = 1:nFiber
        % TTK Tractography
%         x = -fiber.fiber(n).xyzFiberCoord(:,1);
%         y = -fiber.fiber(n).xyzFiberCoord(:,2);
%         z = fiber.fiber(n).xyzFiberCoord(:,3);
%         ijk = affineToIndexT*[x';y';z';ones(1,length(x))];
%         fiber.fiber(n).xyzFiberCoord = [ijk(2,:)',ijk(1,:)',ijk(3,:)'];
%         fiber.fiber(n).xyzFiberCoord = [ijk(2,:)'+1,ijk(1,:)'+1,ijk(3,:)'+1];
        
        % UKF Tractography 
        x = fiber.fiber(n).xyzFiberCoord(:,3);
        y = fiber.fiber(n).xyzFiberCoord(:,2);
        z = fiber.fiber(n).xyzFiberCoord(:,1);
        fiber.fiber(n).xyzFiberCoord = [y+1,x+1,z+1];        
%         if mod(n,50) == 0
%             XYZ = [XYZ;[ijk(2,:)'+1,ijk(1,:)'+1,ijk(3,:)'+1]];
%         end        
    end
    hold on;
    displayFiber(fiber,1:50:nFiber,'default',0,0,'-');            
end



