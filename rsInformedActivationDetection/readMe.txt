-------------------------
Notes
-------------------------
The code in this folder corresponds to the paper:
Ng et al, Connectivity-informed fMRI Activation Detection, MICCAI, 2011


Included in this folder:
- dispParcel.m = Displays the detected parcels on T1
- estSparseInvCov = compute sparse inverse covariance for all subjects
- loadRegressorTask.m = Loads task and motion regressors from gcaSPM.mat
- loadRegressorRest.m = Loads motion regressors from restSPM.mat
- main.m = Calls functions in this folder to find activated parcels
- parcelToAAL.m = Converts detected parcel to AAL ROI labels
- sigTo3D.m = Converts the binary vector "sig", i.e. 1=activated, to 3D matrix for visualization


The folder "old" contains code for smaller scale problem testing.





