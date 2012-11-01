# Batch preprocess IMAGEN data
# Notes: Specify script_path as required to call the python scripts
script_path='/home/bernardyng/code/fMRIanalysis'
for line in `cat /media/GoFlex/research/data/imagen/subjectLists/subjectList.txt`
do
echo $line
cd /media/GoFlex/research/data/imagen/$line/
# Preprocess resting state fMRI data
python $script_path/preprocess.py -tc restfMRI/rest.nii -gm anat/gmMask.nii -wm anat/wmMask.nii -csf anat/csfMask.nii -reg restfMRI/restSPM.txt -dtype 1 -tr 2.2
# Preprocess GCA fMRI data
python $script_path/preprocess.py -tc gcafMRI/gca.nii -gm anat/gmMask.nii -wm anat/wmMask.nii -csf anat/csfMask.nii -reg gcafMRI/gcaSPM.mat -dtype 0 -tr 2.2 -c 10
# Preprocess faces fMRI data
python $script_path/preprocess.py -tc facesfMRI/faces.nii -gm anat/gmMask.nii -wm anat/wmMask.nii -csf anat/csfMask.nii -reg facesfMRI/facesSPM.mat -dtype 0 -tr 2.2 -c 11
done
