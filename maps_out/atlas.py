import nibabel as nib
from nibabel.processing import resample_from_to

# Load your image
img = nib.load("attention_3d_mapped_backprop_cropped_centered_resized.nii.gz")

# Load AAL atlas (reference)
aal = nib.load("aal.nii.gz")  # or aal.maps if from nilearn

# Resample your image into the AAL grid
img_resampled = resample_from_to(img, aal, order=1)  # order=1 = linear interpolation

# Save result
nib.save(img_resampled, "attention_3d_mapped_backprop_atlas_fit.nii.gz")

print("Resampled image shape:", img_resampled.shape)
print("AAL shape:", aal.shape)