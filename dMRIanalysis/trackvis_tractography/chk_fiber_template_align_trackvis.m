% Visualize the fibers to check if aligned to DWI space
% Input:    templatePath = filepath to parcel template
%           fiberPath = filepath to fiber
% Notes:    fibers in matrix space
function chk_fiber_template_align_trackvis(dwiPath,fiberPath)
addpath(genpath('/home/bn228083/code/dMRIanalysis/'));

% Load volume in DWI space
nii = load_untouch_nii(dwiPath);
template = nii.img;
matrixdim = nii.hdr.dime.dim(2:4);
voxdim = nii.hdr.dime.pixdim(2:4);
affineToIndexT = inv([nii.hdr.hist.srow_x;nii.hdr.hist.srow_y;nii.hdr.hist.srow_z;0 0 0 1]);

% Plotting volume
slice = round(matrixdim(3)/2); % Display center axial slice
figure;
surf([1 matrixdim(1)],[1 matrixdim(2)],repmat(slice,[2 2]),uint8(template(:,:,slice)),'facecolor','texture');

% Load fiber
fiber = read_fiber(fiberPath,matrixdim,voxdim);
nFiber = length(fiber.fiber);
for n = 1:nFiber
    % This setting is valid for TrackVis tractography only
    x = -fiber.fiber(n).xyzFiberCoord(:,1);
    y = fiber.fiber(n).xyzFiberCoord(:,2);
    z = fiber.fiber(n).xyzFiberCoord(:,3);
    affineToIndexT(:,4) = 0.4167*affineToIndexT(:,4);
    ijk = affineToIndexT*[x';y';z';ones(1,length(x))];
    fiber.fiber(n).xyzFiberCoord = [ijk(2,:)',ijk(1,:)',ijk(3,:)'];
end
hold on;
display_fiber(fiber,1:5:nFiber,'default',0,0,'-');




