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

subs = 1:10;

for sub = subs
    % Load template in DWI space
    nii = load_untouch_nii([filepath,'data/imagen/',sublist{sub},'/dwi/affparcel500.nii']);
%     nii = load_nii([filepath,'data/imagen/',sublist{sub},'/dwi/affparcel500refined.nii']);
    template = nii.img;
    matrixdim = nii.hdr.dime.dim(2:4);
    voxdim = nii.hdr.dime.pixdim(2:4);
    % Compute transform from world space to index space
    affineToIndexT = inv([nii.hdr.hist.srow_x;nii.hdr.hist.srow_y;nii.hdr.hist.srow_z;0 0 0 1]);
    % Plotting template
    slice = round(matrixdim(3)/2);
    figure;
    surf([1 matrixdim(1)],[1 matrixdim(2)],repmat(slice,[2 2]),uint8(template(:,:,slice)),'facecolor','texture');
%     nii2 = load_untouch_nii([filepath,'data/imagen/',sublist{sub},'/dwi/maskedAveDWI.nii']);
%     affineToIndexD = inv([nii2.hdr.hist.srow_x;nii2.hdr.hist.srow_y;nii2.hdr.hist.srow_z;0 0 0 1]);
    
    % Load fiber
%     fiber = readFiber([filepath,'data/imagen/',sublist{sub},'/dwi/groupFibers.fib'],matrixdim,voxdim);
%     fiber = readFiber([filepath,'data/imagen/',sublist{sub},'/dwi/fibersDense.fib'],matrixdim,voxdim);
%     fiber = readFiber('/media/KINGSTON/viviResults/000003629479/000000112288_meanToOriginal.fib',matrixdim,voxdim);
%     save([filepath,'data/imagen/',sublist{sub},'/dwi/groupFibers.mat'],'fiber')
%     load([filepath,'data/imagen/',sublist{sub},'/dwi/fibersDense.mat']);
    load([filepath,'data/imagen/',sublist{sub},'/dwi/groupFibers.mat']);
    nFiber = length(fiber.fiber);

    XYZ = [];
    % Transform fiber world space coordinates to indices
    for n = 1:nFiber
        x = -fiber.fiber(n).xyzFiberCoord(:,1);
        y = -fiber.fiber(n).xyzFiberCoord(:,2);
        z = fiber.fiber(n).xyzFiberCoord(:,3);
        ijk = affineToIndexT*[x';y';z';ones(1,length(x))];
%         fiber.fiber(n).xyzFiberCoord = [ijk(2,:)',ijk(1,:)',ijk(3,:)'];
        fiber.fiber(n).xyzFiberCoord = [ijk(2,:)'+1,ijk(1,:)'+1,ijk(3,:)'+1];
        if mod(n,50) == 0
            XYZ = [XYZ;[ijk(2,:)'+1,ijk(1,:)'+1,ijk(3,:)'+1]];
        end        
    end
    hold on;
    displayFiber(fiber,1:50:nFiber,'default',0,0,'-');            
end



