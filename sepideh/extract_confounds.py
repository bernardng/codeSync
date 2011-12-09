# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 18:35:59 2011

@author: bernardng

High variance confounds
"""
from nipy.labs import mask as mask_tools
import nibabel


def high_variance_confounds(file_names):
    """ Return confounds time series extracted from voxels with high
        variance.

        Parameters
        ===========
        filenames: list of 3d nifti or analyze files
            The ras (non preprocessed) dataset

        Notes
        ======
        This method is related to what has been published in the
        literature as 'CompCorr' (Behzadi NeuroImage 2007).
    """
    mask = np.ones(nibabel.load(filenames[0]).get_data().shape,
                   dtype=np.bool)
    series, _ = mask_tools.series_from_mask(
                                filenames,
                                mask)
    for serie in series:
        serie[:] = signal.detrend(serie)
    # Retrieve the 1% high variance voxels
    var = np.mean(series**2, axis=-1)
    var_thr = stats.scoreatpercentile(var, 99)
    series = series[var > var_thr]
    u, s, v = linalg.svd(series, full_matrices=False)
    v = v[:10]
    return v
