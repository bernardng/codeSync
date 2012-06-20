# Batch preprocess IMAGEN data
# Notes: - Add dMRIanalysis folder to PATH in ~./bashrc
#        - path-to-folder is required to call the python scripts
script_path='/home/bn228083/code/dMRIanalysis/'
for line in `cat subjectListDWI.txt`
do
cd /volatile/bernardng/data/imagen/$line/dwi/
rm dwiECC.ecclog # rotation tables from previous run are concatenated if dwiECC.ecclog exists
# Eddy current correction
fsl4.1-eddy_correct dwi.nii dwi_ecc.nii 0
gunzip -f dwi_ecc.nii.gz
# Tensor reorientation
$script_path/rotatebvecs bvec.bvec bvec_ecc.bvec dwi_ecc.ecclog
# Convert gradient table to three columns
python $script_path/columnizebvecs.py -i bvec_ecc.bvec
done


