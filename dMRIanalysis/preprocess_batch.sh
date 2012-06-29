# Batch preprocess IMAGEN data
# Notes: Specify script_path as required to call the python scripts
script_path='/home/bn228083/code/dMRIanalysis/'
for line in `cat /volatile/bernardng/data/imagen/subjectLists/subjectListDWI.txt`
do
cd /volatile/bernardng/data/imagen/$line/dwi/
rm dwiECC.ecclog # rotation tables from previous run are concatenated if dwiECC.ecclog exists
# Eddy current correction
fsl4.1-eddy_correct dwi.nii dwi_ecc.nii 0
gunzip -f dwi_ecc.nii.gz
# Tensor reorientation
$script_path/rotatebvecs bvec.bvec bvec_ecc.bvec dwi_ecc.ecclog
# Convert gradient and b-value tables to columns
python $script_path/columnize.py -i bvec_ecc.bvec -o bvec_ecc.txt -d 6
python $script_path/columnize.py -i bval.bval -o bval.txt -d 0
done


