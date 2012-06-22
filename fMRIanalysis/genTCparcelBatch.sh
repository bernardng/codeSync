# Batch preprocess IMAGEN data
# Notes: Specify script_path as required to call the python scripts
script_path='/home/bn228083/code/fMRIanalysis'
for line in `cat /volatile/bernardng/data/imagen/subjectLists/subjectList.txt`
do
echo $line
cd /volatile/bernardng/data/imagen/$line/
# Generate parcel time courses for resting state fMRI data
python $script_path/gen_tc_parcel.py -tc restfMRI/tc_vox.mat -gm anat/gmMask.nii -wm anat/wmMask.nii -csf anat/csfMask.nii -t /volatile/bernardng/data/imagen/group/fs_parcel500.nii
# Generate parcel time courses for GCA fMRI data
python $script_path/gen_tc_parcel.py -tc gcafMRI/tc_vox.mat -gm anat/gmMask.nii -wm anat/wmMask.nii -csf anat/csfMask.nii -t /volatile/bernardng/data/imagen/group/fs_parcel500.nii
# Generate parcel time courses for faces fMRI data
python $script_path/gen_tc_parcel.py -tc facesfMRI/tc_vox.mat -gm anat/gmMask.nii -wm anat/wmMask.nii -csf anat/csfMask.nii -t /volatile/bernardng/data/imagen/group/fs_parcel500.nii
done

