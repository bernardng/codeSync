"""
Exemplar script for renaming files
"""
import os
import glob

for folder in glob.glob('0*'):
    for filepath in glob.glob(os.path.join(folder, 'facesfMRI', '*_task_*')):
        new_path = filepath.replace('_task', '')
        os.rename(filepath, new_path)

    
    
    
        