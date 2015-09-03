import os.path as op
import time
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import dipy.tracking.streamline as dts
import dipy.tracking.utils as dtu
import dipy.data as dpd

import nibabel as nib
import dipy.data as dpd
import os
import dipy.data as dpd

from dipy.align import VerbosityLevels
from dipy.align.metrics import CCMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from scipy.ndimage.morphology import binary_dilation

use_mayavi = False
if use_mayavi:
    import mayavi
    import mayavi.mlab as mlab
    from tvtk.tools import visual
    from tvtk.api import tvtk


ni, gtab = dpd.read_stanford_hardi()
hardi_data = ni.get_data()
hardi_affine = ni.get_affine()
b0 = hardi_data[..., gtab.b0s_mask]
mean_b0 = np.mean(b0, -1)

ni_b0 = nib.Nifti1Image(mean_b0, hardi_affine)
ni_b0.to_filename('mean_b0.nii')
plt.matshow(mean_b0[:,:,mean_b0.shape[-1]//2], cmap=cm.bone)

MNI_T2 = dpd.read_mni_template()
MNI_T2_data = MNI_T2.get_data()
MNI_T2_affine = MNI_T2.get_affine()

level_iters = [10, 10, 5]
dim = 3
metric = CCMetric(dim)
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, step_length=0.25)
sdr.verbosity = VerbosityLevels.DIAGNOSE
mapping = sdr.optimize(MNI_T2_data, mean_b0, MNI_T2_affine, hardi_affine)
warped_b0 = mapping.transform(mean_b0)
plt.matshow(warped_b0[:,:,warped_b0.shape[-1]//2], cmap=cm.bone)
plt.matshow(MNI_T2_data[:, :, MNI_T2_data.shape[-1]//2], cmap=cm.bone)

new_ni = nib.Nifti1Image(warped_b0, MNI_T2_affine)
new_ni.to_filename('./warped_b0.nii.gz')

afqpath = 'D:/opt/AFQ/'
LOCC_ni = nib.load(os.path.join(afqpath,'templates/callosum2/L_Occipital.nii.gz'))
ROCC_ni = nib.load(os.path.join(afqpath,'templates/callosum2/R_Occipital.nii.gz'))
midsag_ni = nib.load(os.path.join(afqpath,'templates/callosum2/Callosum_midsag.nii.gz'))
LOCC_data = LOCC_ni.get_data()
ROCC_data = ROCC_ni.get_data()
midsag_data = midsag_ni.get_data()

midsag_affine = midsag_ni.get_affine()
midsag_data = midsag_ni.get_data()
warped_midsag = mapping.transform_inverse(midsag_data)

# Dilate along the x axis
structure = np.zeros((3,3,3))
structure[:3,1,1] = 1
dilated = binary_dilation(midsag_data, structure)

if use_mayavi:
    # Show the actual voxels that form the mask
    X, Y, Z = [],[],[]
    for x in range(dilated.shape[0]):
        for y in range(dilated.shape[1]):
            for z in range(dilated.shape[2]):
                if dilated[x,y,z]:
                    X.append(x)
                    Y.append(y)
                    Z.append(z)                
    mlab.points3d(X,Y,Z, color=(1,1,1), scale_factor=0.9)

# Do the same process with the dilated mask
warped_midsag = mapping.transform_inverse(dilated)
im = np.sum(np.ceil(dilated), axis=0)
plt.matshow(im)

