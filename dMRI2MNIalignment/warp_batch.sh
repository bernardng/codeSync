# Batch warp IMAGEN data
# Notes: Specify script_path as required to call the python scripts
#	 T2 template must be resampled to DWI data space prior to registration
script_path=/home/bernardyng/code/dMRI2MNIalignment
for line in `cat /media/GoFlex/research/data/imagen/subjectLists/subjectListDWI.txt`
#for line in `cat /media/GoFlex/research/data/imagen/group/groupFiber/subjectListDWITemp.txt`
do
echo "subject"$line
cd /media/GoFlex/research/data/imagen/$line/dwi/warp_mni/
# Extract average B0 volume
#python $script_path/extract_B0.py -i dwi_ecc.nii -b bval.bval
# Perform skull stripping
#fsl4.1-bet b0.nii b0_ss.nii -f 0.25
# Affinely align B0 to T2 template
#fsl4.1-flirt -in b0.nii -ref /media/GoFlex/research/data/imagen/group/groupFiber/t2_mni.nii -omat affine.txt
# Reorient bvec_ecc.bvec
#python $script_path/reorient_bvec.py -i bvec_ecc.bvec -a affine.txt
# Create column version of bvec_ecc_reorient.bvec
#python $script_path/columnize.py -i bvec_ecc_reorient.bvec -o bvec_ecc_reorient.txt -d 6
# Nonrigidly warp B0 volume to T2 template
#fsl4.1-fnirt --in=b0.nii --ref=/media/GoFlex/research/data/imagen/group/groupFiber/t2_mni.nii --subsamp=8,4,2,2 --cout=d_field.nii --aff=affine.txt
# Apply warp to DWI volumes
#fsl4.1-applywarp -i dwi_ecc.nii -o dwi_ecc_mni.nii -r /media/GoFlex/research/data/imagen/group/groupFiber/t2_mni.nii -w d_field.nii.gz
#gunzip -f dwi_ecc_mni.nii.gz
done
