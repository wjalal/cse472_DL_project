import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation

# Paths
input_nii_path = "adni_storage/ADNI_nii_gz_bias_corrected/I1558920_035_S_7050.stripped.N4.nii.gz"  # Replace with your input file path
output_nii_path = "sitk_I1558920_035_S_7050.stripped.N4.nii.gz"  # Replace with desired output path

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

# Step 2: Convert to SimpleITK Image
sitk_img = sitk.GetImageFromArray(data)
sitk_img.SetSpacing([affine[0, 0], affine[1, 1], affine[2, 2]])

# Step 3: Resample the Volume to Target Number of Slices
original_size = sitk_img.GetSize()  # (width, height, depth)
original_spacing = sitk_img.GetSpacing()  # (spacing_x, spacing_y, spacing_z)

# Calculate new spacing
new_spacing = list(original_spacing)
new_spacing[2] = (original_spacing[2] * original_size[2]) / target_slices

# Define new size
new_size = [original_size[0], original_size[1], target_slices]

# Resample using SimpleITK
resampler = sitk.ResampleImageFilter()
resampler.SetOutputSpacing(new_spacing)
resampler.SetSize(new_size)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetOutputDirection(sitk_img.GetDirection())
resampler.SetOutputOrigin(sitk_img.GetOrigin())
resampled_img = resampler.Execute(sitk_img)

# Step 4: Save the Resampled Image
resampled_data = sitk.GetArrayFromImage(resampled_img)
print (resampled_data.shape)
# Convert back to Nibabel NIfTI format
resampled_nii = nib.Nifti1Image(resampled_data, affine=nii_img.affine)
nib.save(resampled_nii, output_nii_path)

print(f"Resampled image saved to {output_nii_path}")