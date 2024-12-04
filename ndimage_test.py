import numpy as np
from scipy.ndimage import zoom

def resample_volume_with_scipy(data, target_slices=160, original_spacing=None):
    """
    Resample a 3D volume (e.g., MRI or CT) to a fixed number of slices in the depth dimension using SciPy,
    while maintaining voxel aspect ratio.
    
    Parameters:
    - data: numpy.ndarray
        The input 3D volume as a NumPy array with shape (D, H, W).
    - target_slices: int
        The desired number of slices in the depth (first) dimension.
    - original_spacing: tuple or list
        The original spacing of the volume (x, y, z), in millimeters.
    
    Returns:
    - resampled_data: numpy.ndarray
        The resampled 3D volume with shape (target_slices, H, W).
    """
    # Ensure original_spacing is provided
    if original_spacing is None:
        raise ValueError("original_spacing must be provided for spacing adjustment.")
    
    # Current depth of the volume
    original_depth = data.shape[0]
    
    # Compute scaling factor for the depth dimension
    depth_scale = target_slices / original_depth
    
    # Compute the scaling factors for each axis (depth, height, width)
    scaling_factors = [depth_scale, original_spacing[1] / original_spacing[1], original_spacing[2] / original_spacing[2]]
    
    # Resample the volume with the correct scaling
    resampled_data = zoom(data, scaling_factors, order=3)  # Use cubic interpolation
    
    return resampled_data

import nibabel as nib
import numpy as np
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation

# Paths
input_nii_path = "adni_storage/ADNI_nii_gz_bias_corrected/I1558920_035_S_7050.stripped.N4.nii.gz"  # Replace with your input file path
output_nii_path = "zoom_I1558920_035_S_7050.stripped.N4.nii.gz"  # Replace with desired output path

# Parameters
target_slices = 160  # Target number of slices in the depth dimension

# Step 1: Load and Reorient with Nibabel
nii_img = nib.load(input_nii_path)

# Reorient to RAS
orig_ornt = io_orientation(nii_img.affine)
ras_ornt = axcodes2ornt(("R", "A", "S"))
ornt_trans = ornt_transform(orig_ornt, ras_ornt)

data = apply_orientation(nii_img.get_fdata(), ornt_trans)
print (data.shape)
affine = nii_img.affine  # Affine transformation matrix after reorientation
spacing = np.abs(np.diag(affine)[:3])  # Get the first three values (x, y, z voxel size)

resampled_data = resample_volume_with_scipy(data, 160, spacing)
print (resampled_data.shape)
# Convert back to Nibabel NIfTI format
resampled_nii = nib.Nifti1Image(resampled_data, affine=nii_img.affine)
nib.save(resampled_nii, output_nii_path)

print(f"Resampled image saved to {output_nii_path}")