% Counting #fibers going through each pair of ROIs
clear all; close all;
filepath = 'I:/research/data/imagen/';
addpath(genpath('I:/research/code/dMRIanalysis/'));

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

dwiPath = [filepath,'group/groupFiber/t2_mni.nii'];%ica_roi_parcel150_refined.nii'];
fiberPath = [filepath,'group/groupFiber/group',grp,'/results_',trackAlgor,'/tracks_',trackAlgor,'.fib'];
eval(['chk_fiber_template_align_',trackAlgor,'(dwiPath,fiberPath)']);




