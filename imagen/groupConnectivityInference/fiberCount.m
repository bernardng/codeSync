% Counting #fibers going through each pair of ROIs
addpath(genpath('I:/research/code/dMRIanalysis/'));
filepath = 'I:/research/data/imagen/';

% Select tractography algorithm
whichAlgor = 3; % 1 = UKF, 2 = TTK, 3 = TrackVis
if whichAlgor == 1
    trackAlgor = 'ukf';
elseif whichAlgor == 2
    trackAlgor = 'ttk';
elseif whichAlgor == 3
    trackAlgor = 'trackvis';
end

grp = '_all';

templatePath = [filepath,'group/ica_roi_parcel500_refined_2mm.nii'];
fiberPath = [filepath,'group/groupFiber/group',grp,'/results_',trackAlgor,'/tracks_',trackAlgor,'.fib'];
connPath = [filepath,'group/groupFiber/group',grp,'/results_',trackAlgor,'/K_gaussian_blur_ica_roi_parcel500_refined.mat'];
eval(['fiber_count_',trackAlgor,'_gaussian_blur(templatePath,fiberPath,connPath);']);

