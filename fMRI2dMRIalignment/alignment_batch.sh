# Batch align WM mask and parcel template to DWI
# Notes: Specify script_path as required to call the python scripts
script_path='/home/bn228083/code/fMRI2dMRIalignment'
for line in `cat /volatile/bernardng/data/imagen/subjectLists/subjectListDWI.txt`
do
echo $line
cd /volatile/bernardng/data/imagen/$line/dwi/
cp /volatile/bernardng/data/imagen/$line/restfMRI/rest.nii .
cp /volatile/bernardng/data/imagen/group/fs_parcel500.nii .
cp /volatile/bernardng/data/imagen/$line/anat/gmMask.nii .
cp /volatile/bernardng/data/imagen/$line/anat/wmMask.nii .
cp /volatile/bernardng/data/imagen/$line/anat/csfMask.nii .

# Generate average volumes
python $script_path/gen_ave_vol.py -i rest.nii
python $script_path/gen_ave_vol.py -i dwi_ecc.nii

# Skull strip the average volumes
fsl4.1-bet rest_ave.nii rest_ave_ss.nii -m -f 0.3
gunzip -f rest_ave_ss.nii.gz
gunzip -f rest_ave_ss_mask.nii.gz
fsl4.1-bet dwi_ecc_ave.nii dwi_ecc_ave_ss.nii -m -f 0.3
gunzip -f dwi_ecc_ave_ss.nii.gz
gunzip -f dwi_ecc_ave_ss_mask.nii.gz

# Resample to DWI resolution
python $script_path/resample.py -i rest_ave_ss.nii -r dwi_ecc_ave_ss.nii
python $script_path/resample.py -i fs_parcel500.nii -r dwi_ecc_ave_ss.nii -t 1
python $script_path/resample.py -i gmMask.nii -r dwi_ecc_ave_ss.nii
python $script_path/resample.py -i wmMask.nii -r dwi_ecc_ave_ss.nii
python $script_path/resample.py -i csfMask.nii -r dwi_ecc_ave_ss.nii

# Align EPI to DWI volume
fsl4.1-flirt -in rest_ave_ss_rs.nii -ref dwi_ecc_ave_ss.nii -out rest_ave_ss_rs_aff.nii -omat mni3mmtodwi_aff.txt 
gunzip -f rest_ave_ss_rs_aff.nii.gz

# Apply learned warp to GM, mask, WM mask, CSF mask, and parcel template to DWI
fsl4.1-flirt -in fs_parcel500_rs.nii -ref dwi_ecc_ave_ss.nii -applyxfm -init mni3mmtodwi_aff.txt -out fs_parcel500_rs_aff.nii -interp nearestneighbour
gunzip -f fs_parcel500_rs_aff.nii.gz
fsl4.1-flirt -in gmMask_rs.nii -ref dwi_ecc_ave_ss.nii -applyxfm -init mni3mmtodwi_aff.txt -out gmMask_rs_aff.nii
gunzip -f gmMask_rs_aff.nii.gz
fsl4.1-flirt -in wmMask_rs.nii -ref dwi_ecc_ave_ss.nii -applyxfm -init mni3mmtodwi_aff.txt -out wmMask_rs_aff.nii
gunzip -f wmMask_rs_aff.nii.gz
fsl4.1-flirt -in csfMask_rs.nii -ref dwi_ecc_ave_ss.nii -applyxfm -init mni3mmtodwi_aff.txt -out csfMask_rs_aff.nii
gunzip -f csfMask_rs_aff.nii.gz

# Move results to alignment folder
mkdir -p alignment
mv res*.nii alignment
mv dwi_ecc_*.nii alignment
mv *Mas*.nii alignment
mv alignment/*Mask_rs_aff.nii .
mv fs_parcel50*.nii alignment
mv alignment/fs_parcel500_rs_aff.nii .
mv mni3mmtodwi_aff.txt alignment
done


