# Batch align WM mask and parcel template to DWI
# Notes: Specify script_path as required to call the python scripts
#	 Might need to enter FSL environment by typing "fsl"; rem to type "exit" when done
script_path='/home/bernardyng/code/fMRI2dMRIalignment'
for line in `cat /media/GoFlex/research/data/imagen/subjectLists/subjectListDWI.txt`
do
echo $line
cd /media/GoFlex/research/data/imagen/$line/dwi/
#cp /media/GoFlex/research/data/imagen/$line/restfMRI/rest.nii .
#cp /media/GoFlex/research/data/imagen/group/fs_parcel500.nii .
#cp /media/GoFlex/research/data/imagen/$line/anat/wmMask.nii .

# Generate average volumes
#python $script_path/gen_ave_vol.py -i rest.nii
#python $script_path/gen_ave_vol.py -i dwi_ecc.nii

# Skull strip the average volumes
#fsl4.1-bet rest_ave.nii rest_ave_ss.nii -m -f 0.3
#gunzip -f rest_ave_ss.nii.gz
#gunzip -f rest_ave_ss_mask.nii.gz
#fsl4.1-bet dwi_ecc_ave.nii dwi_ecc_ave_ss.nii -m -f 0.3
#gunzip -f dwi_ecc_ave_ss.nii.gz
#gunzip -f dwi_ecc_ave_ss_mask.nii.gz

# Resample wmMast to EPI resolution
#python $script_path/resample.py -i wmMask.nii -r rest_ave_ss.nii

# Align EPI to DWI volume
#fsl4.1-flirt -in rest_ave_ss.nii -ref dwi_ecc_ave_ss.nii -out rest_ave_ss_aff.nii -omat mni3mmtodwi_aff.txt 
#gunzip -f rest_ave_ss_aff.nii.gz

# Apply learned warp to WM mask and parcel template to DWI
fsl4.1-flirt -in /media/GoFlex/research/data/imagen/group/fs_w_cer_parcel500_no_zero_tc.nii -ref alignment/dwi_ecc_ave_ss.nii -applyxfm -init alignment/mni3mmtodwi_aff.txt -out fs_w_cer_parcel500_no_zero_tc_aff.nii -interp nearestneighbour
gunzip -f fs_w_cer_parcel500_no_zero_tc_aff.nii.gz
#fsl4.1-flirt -in wmMask_rs.nii -ref dwi_ecc_ave_ss.nii -applyxfm -init mni3mmtodwi_aff.txt -out wmMask_rs_aff.nii
#gunzip -f wmMask_rs_aff.nii.gz

# Binarize the WM mask
#python $script_path/gen_bin_mask.py -i wmMask_rs_aff.nii -t 0.3

# Move results to alignment folder
#mkdir -p alignment
#mv res*.nii alignment
#mv dwi_ecc_*.nii alignment
#mv *Mas*.nii alignment
#mv alignment/*Mask_rs_aff.nii .
#mv alignment/*Mask_rs_aff_bin.nii .
#mv fs_parcel50*.nii alignment
#mv alignment/fs_parcel500_aff.nii .
#mv mni3mmtodwi_aff.txt alignment
done


