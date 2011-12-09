############################################################################
# Preprocessing sepideh's data
# Load data
# Extract GM voxel time courses
# Regress out motion artifacts and WM signal
# Bandpass filter
# Note: TR = 2.4 sec
############################################################################
import os
import glob
import pylab as pl
import numpy as np
import parietal
from parietal.time_series import preprocessing
from scipy import stats
from scipy import signal
from scipy import linalg
from nipy.labs import mask as mask_utils
from nipy.labs import as_volume_img
from nipy.labs import viz

BASE_DIR = "/volatile/bernardng/data/sepideh/testSubject"
# Path to warped greymatter probabilistic mask
gm_file = os.path.join(BASE_DIR, "t1mri", "wc1sga070108233-0012-00001-000160-01.hdr")
# Path to randomly chosen EPI volume as reference
epi_ref_file = os.path.join(BASE_DIR, "fmri", "swfga070108233-0004-00815-000815-01.hdr")
# Paths to all EPI volumes with name matching wildcard
epi_files = sorted(glob.glob(os.path.join(BASE_DIR, 'fmri', 'swf*.hdr')))
# Load gm_img and epi_img objects
gm_img = as_volume_img(gm_file)
epi_ref_img = as_volume_img(epi_ref_file)
# Resample tissue mask to grid of epi
gm_img = gm_img.resampled_to_img(epi_ref_img)
# Extract tissue mask
gm = gm_img.get_data()
# Normalize the mask to [0,1]
gm -= gm.min()
gm /= gm.max()
# Threshold tissue mask
gm_mask = (gm > .5)
# Find largest connected component
gm_mask = mask_utils.largest_cc(gm_mask)

# Extract graymatter voxel timecourses
time_series_gm, header_gm = mask_utils.series_from_mask(epi_files, gm_mask)
time_series_gm = preprocessing.standardize(time_series_gm).T
n_tpts = time_series_gm.shape[0]
# Load motion regressors
motion_regressor = np.loadtxt(os.path.join(BASE_DIR, "fmri", "rp_fga070108233-0004-00002-000002-01.txt"))
motion_regressor = preprocessing.standardize(motion_regressor)
# Bandpass filter to remove DC offset and low frequency drifts
beta, _, _, _ = linalg.lstsq(motion_regressor,time_series_gm)
time_series_gm -= np.dot(motion_regressor,beta)
f_cut = np.array([0.01, 0.1])
tr = 2.4
samp_freq = 1 / tr 
w_cut = f_cut * 2 / samp_freq
b, a = signal.butter(5, w_cut, btype='bandpass', analog=0, output='ba')
for tc in time_series_gm.T:
    tc[:] = signal.filtfilt(b, a, tc) # Modify timer_series_gm in place

# Mask out non-graymatter voxel in ROI mask
roi_mask_file = "/volatile/bernardng/templates/yeoAndBuckner/YeoMNI152Refined/Yeo2011_17Networks_MNI152_FreeSurferConformed1mmRefined.nii"
roi_mask_img = as_volume_img(roi_mask_file)
roi_mask_img = roi_mask_img.resampled_to_img(epi_ref_img, interpolation='nearest')
roi_mask = roi_mask_img.get_data()
roi_mask = roi_mask[np.nonzero(gm_mask)]
roi = np.unique(roi_mask)
n_roi = np.unique(roi).shape[0] - 1
tc_roi = np.zeros([n_tpts, n_roi])
for i in np.arange(n_roi) + 1: # Skip background
    tc_roi[:,i-1] = time_series_gm[:,roi_mask == roi[i]].sum(1)
    
# Change the directory   
#np.save(os.path.join(BASE_DIR, "fmri", "tc_gm"),time_series_gm)

############################################################################
# Plotting
############################################################################
#slicer = viz.plot_anat(gm_img.get_data(), gm_img.affine, figure=1)
#epi = epi_img.get_data()
#affine = epi_img.affine
#slicer.plot_map(epi, affine, threshold=stats.scoreatpercentile(epi.ravel(), 90))