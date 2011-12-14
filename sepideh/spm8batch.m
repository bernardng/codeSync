% An example of multi- subject & session SPM5 batch script for preprocessing
%
% Assume the following data organization:
%   Experiment
%       |_____ subject1
%       |         |_____ structural
%       |         |_____ functional
%       |                    |_____ session1
%       |                    |_____ session2
%       |                    |_____ ...
%       |_____ subject2
%       |_____  ...
%
% This will perform the following steps on a subject basis:
%  * Realign:    estimate and reslice (mean only)
%  * Coregister: estimate only (structural -> mean functional)
%  * Segment
%  * Normalize:  write
%  * Smooth
% Then the mean normalized structural image will be created.

%% Experiment folder
data_path = '/volatile/bernardng/data/sepideh/raw/';

%% Subjects folders
subjects = {'MP070143' 'MZ080021' 'MB080023' 'SV080025' 'CB080029' 'CG080027' 'MP080022'};
%[unused,d] = spm_select('List',data_path,'');
%d = cellstr(d); d = d(strmatch('s',d));

%% Sessions folders
sessions = {'rsfMRI'};
%sessions = cellfun(@(x) sprintf('session%d',x),{1 2 3},'UniformOutput',false);
%clear sessions; for i=[1 2 3], sessions{i} = sprintf('session%d',i); end

%% Set Matlab path
%--------------------------------------------------------------------------
addpath('/volatile/bernardng/matlabToolboxes/spm8');   % SPM path including editfilename.m

%% Initialise SPM defaults
%--------------------------------------------------------------------------
spm('Defaults','fMRI');
spm_jobman('initcfg');

%% Add path to nifti tools
addpath(genpath('/volatile/bernardng/matlabToolboxes/nifti'));

%% Predefine the number of volumes
nVols = 821;

matlabpool(6);
%% Loop over subjects
%--------------------------------------------------------------------------
parfor i=1:numel(subjects)
    fmriPath = [data_path,subjects{i},'/',sessions{1},'/'];
    reslice_nii([fmriPath,'bold_audiospont_1.nii'],[fmriPath,'bold_audiospont_1']); % Save as .hdr/.img required for SPM8
    expand_nii_scan([fmriPath,'bold_audiospont_1.hdr'],1:nVols,[]);
    % Remove redundant files
    delete([fmriPath,'bold_audiospont_1.hdr']);
    delete([fmriPath,'bold_audiospont_1.img']);
    delete([fmriPath,'bold_audiospont_1.mat']);
    
%     clear jobs a f 
    f = cell(numel(sessions),1);
    for j=1:numel(sessions)
        f{j} = spm_select('FPList', fullfile(data_path,subjects{i},sessions{j}), '^bold.*\.img$');
    end
    a = spm_select('FPList', fullfile(data_path,subjects{i},'anat'), '^anat.*\.img$');
    
    fprintf('Preprocessing subject "%s" (%s)\n',subjects{i},sprintf('%d ',cellfun(@(x) size(x,1),f)));
    
    jobs = cell(2,1);
    %% CHANGE WORKING DIRECTORY (useful for .ps only)
    %----------------------------------------------------------------------
    jobs{1}.util{1}.cdir.directory = cellstr(fullfile(data_path,subjects{i}));
    
    %% REALIGN: ESTIMATE AND RESLICE
    %----------------------------------------------------------------------
    for j=1:numel(sessions)
        jobs{2}.spatial{1}.realign{1}.estwrite.data{j} = cellstr(f{j});
    end
%     jobs{2}.spatial{1}.realign{1}.estwrite.roptions.which = [0 1]; % mean image only
    
    %% COREGISTER: structural -> mean functional
    %----------------------------------------------------------------------
    jobs{2}.spatial{2}.coreg{1}.estimate.ref = editfilenames(f{1}(1,:),'prefix','mean');
    jobs{2}.spatial{2}.coreg{1}.estimate.source = cellstr(a);
    
    %% SEGMENT
    %----------------------------------------------------------------------
    jobs{2}.spatial{3}.preproc.data = cellstr(a);
    jobs{2}.spatial{3}.preproc.output.GM     = [0 1 1];   % gray modulated normalized
    jobs{2}.spatial{3}.preproc.output.WM     = [0 1 1];   % white modulated normalized  
    jobs{2}.spatial{3}.preproc.output.CSF    = [0 1 1];   % csf modulated normalized

    %% NORMALISE: WRITE: anatomy and functional with <> resolutions
    %----------------------------------------------------------------------
    jobs{2}.spatial{4}.normalise{1}.write.subj.matname  = editfilenames(a,'suffix','_seg_sn','ext','.mat');
    jobs{2}.spatial{4}.normalise{1}.write.subj.resample = cellstr(char(f));
    jobs{2}.spatial{4}.normalise{1}.write.roptions.vox  = [3 3 3];

    jobs{2}.spatial{4}.normalise{2}.write.subj.matname  = editfilenames(a,'suffix','_seg_sn','ext','.mat');
    jobs{2}.spatial{4}.normalise{2}.write.subj.resample = editfilenames(a,'prefix','m');
    jobs{2}.spatial{4}.normalise{2}.write.roptions.vox  = [1 1 1];
    
%     %% SMOOTH
%     %----------------------------------------------------------------------
%     jobs{2}.spatial{5}.smooth.data = editfilenames(char(f),'prefix','w');
%     jobs{2}.spatial{5}.smooth.fwhm = [10 10 10];
    
    %% SAVE AND RUN JOB
    %----------------------------------------------------------------------
%     save(fullfile(data_path,subjects{i},'job_preprocessing.mat'),'jobs');
%     spm_jobman('interactive',jobs);
    spm_jobman('run',jobs);
    
    collapse_nii_scan('wbold*.img','wbold_audiospont_1',fmriPath); % Save warped 3D fMRI volumes into 4D .hdr/.img pair
end