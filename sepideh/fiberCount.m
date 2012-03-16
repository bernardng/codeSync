% Counting #fibers going through each pair of ROIs
filepath = '/volatile/bernardng/';
addpath(genpath([filepath,'matlabToolboxes/dwiUtils']));
addpath(genpath([filepath,'matlabToolboxes/nifti']));
currPath = pwd;
dataPath = [filepath,'/data/sepideh/preprocessedBN/'];
subjects = {'MP070143' 'MZ080021' 'MB080023' 'SV080025' 'CB080029' 'CG080027' 'MP080022'};
nSubs = length(subjects);
A = [-1.8750 0 0 118.1250;0 -1.8750 0 120;0 0 2 -60;0 0 0 1]; % Affine transform to world space obtained using itkImageInfo
for sub = 1:nSubs
    cd([dataPath,subjects{sub},'/diffusionMRI/']);
    
    % Put in the warped version if successful
    nii = load_nii('roiTemplateflip.nii');
    
    roi = nii.img;
    fiber = readFiber('fibers.fib',matrixdim,voxdim);
    nFiber = length(fiber.fiber);
    fiberProb = zeros(size(roi));
    for n = 1:nFiber
        x = fiber.fiber(n).xyzFiberCoord(:,1);
        y = fiber.fiber(n).xyzFiberCoord(:,2);
        z = fiber.fiber(n).xyzFiberCoord(:,3);
        ijk = inv(A)*[x';y';z';ones(1,length(x))];
       
    end
            
end
cd(currPath);