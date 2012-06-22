# Batch UKF fiber tracking of IMAGEN data
# Notes: Specify script_path as required to call the python scripts
script_path='/home/bn228083/code/dMRIanalysis'
for line in `cat /volatile/bernardng/data/imagen/subjectLists/subjectListDWI.txt`
do
cd /volatile/bernardng/data/imagen/$line/dwi/
python $script_path/convertType.py -i dwi_ecc.nii -t 1
python $script_path/dwi_nii2nrrd.py -i dwi_ecc_int16.nii
python $script_path/gen_bin_mask.py -i wmMask_rs_aff.nii -t 0.33
python $script_path/nii2nrrd.py -i wmMask_rs_aff_bin.nii
UKFTractography --dwiFile dwi_ecc_int16.nhdr --maskFile wmMask_rs_aff_bin.nhdr --tracts tracks_ukf.vtk --numTensor 2 --noTransformPosition --seedsFile wmMask_rs_aff_bin.nhdr --seedsPerVoxel 5 --numThreads 7
# Move results to results_ukf folder
mkdir -p results_ukf
mv dwi_ecc_int* results_ukf
mv wmMask* alignment
mv alignment/wmMask_rs_aff_bin.nii .
mv tracks_ukf.vtk results_ukf
done
