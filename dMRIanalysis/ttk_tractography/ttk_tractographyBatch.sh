# Batch UKF fiber tracking of IMAGEN data
# Notes: Specify script_path as required to call the python scripts
script_path='/home/bn228083/code/dMRIanalysis'
for line in `cat /volatile/bernardng/data/imagen/subjectLists/subjectListDWI.txt`
do
cd /volatile/bernardng/data/imagen/$line/dwi/
ttk-utils apply_mask -i dwi_ecc.nii -m wmMask_rs_aff_bin.nii -o dwi_ecc_masked.nii -t 0
ttk estimate -i dwi_ecc_masked.nii -g bvec_ecc.txt -b 0 -r 1 -o tensor.nii
ttk tractography -i tensor.nii -fa1 0.2 -fa2 0.1 -s 1.0 -samp 1 -fs 1.0 -t 0.5 -m 2 -min 10 -l -n 7 -o tracks_ttk.fib

# Move results to results_ttk folder
mkdir -p results_ttk
mv dwi_ecc_masked.nii results_ttk
mv tensor.nii results_ttk
mv tracks_ttk.fib results_ttk
done
