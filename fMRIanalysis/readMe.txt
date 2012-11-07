Required folder structure
----------------------------

Below is an example for the IMAGEN database

.../data/imagen/subject#/
.../data/imagen/subject#/anat/
.../data/imagen/subject#/dwi/
.../data/imagen/subject#/facesfMRI/
.../data/imagen/subject#/gcafMRI/
.../data/imagen/subject#/multimodalConn/
.../data/imagen/subject#/restfMRI/
.../data/imagen/group/
.../data/imagen/subjectLists/


Combining predefined ROIs with remaining graymatter voxels
-------------------------------------------------------------
1. Combine the ROIs and GM voxels into a single volume.
brain = np.zeros(np.shape(gm))
brain[gm > 0] = 1
brain[roi > 1] = roi[roi > 1]

2. Remove disconnected voxels in roi and put into gm using relabel_disconnected_rois.py

3. Extract gray matter voxels, find disconnected voxels, and remove from brain
gm = brain == 1
labels, n_labels = label(gm, structure=generate_binary_structure(3,1))
brain[labels > 1] = 0 # Assume labels == 1 is the largest connected component, and the rest are isolated voxels 

4. Apply Ward and remove any remaining isolated voxels, e.g. clusters < 6 voxels.



