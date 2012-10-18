import os
import numpy as np

BASE_DIR = "/media/GoFlex/research/data/imagen"
#subjectList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/subjectListDWI.txt"), dtype='str')
subjectList = np.loadtxt(os.path.join(BASE_DIR, "group/groupFiber/goodWarpSubjectListDWI.txt"), dtype='str')

i = 0
for sub in subjectList[0:5]:
    bvec = np.loadtxt(os.path.join(BASE_DIR, sub, "dwi/warp_mni/bvec_ecc_reorient.txt"))
    bval = np.loadtxt(os.path.join(BASE_DIR, sub, "dwi/warp_mni/bval.txt"))
    if i == 0:
        bvec_group = bvec.copy()
        bval_group = bval.copy()
    else:
        bvec_group = np.vstack((bvec_group, bvec))
        bval_group = np.hstack((bval_group, bval))
    i = i + 1
np.savetxt(os.path.join(BASE_DIR, "group/groupFiber/bvec_ecc_reorient_group.txt"), bvec_group, fmt='%1.6f')
np.savetxt(os.path.join(BASE_DIR, "group/groupFiber/bval_group.txt"), bval_group, fmt='%1.0f')
