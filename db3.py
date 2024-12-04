from deepbrain import Extractor
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
# Load the NIfTI image
filename = "ADNI/ADNI_ADNI_12M4_TS_2_20061213085558_2.nii"
nii_img = nib.load(filename)
data = nii_img.get_fdata()

# Initialize the extractor
ext = Extractor()

# Run the extractor and get the mask
prob = ext.run(data)
brain_mask = prob > 0.5  # Threshold for mask

# Apply the mask to extract the brain
brain_only = data * brain_mask

# Save the brain-extracted image
brain_nii = nib.Nifti1Image(brain_only, nii_img.affine, nii_img.header)
nib.save(brain_nii, "output_brain2.nii")

# Visualize the results
slice_idx = data.shape[2] // 2
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(data[:, :, slice_idx], cmap="gray")
plt.title("Original Image")
plt.subplot(1, 3, 2)
plt.imshow(brain_mask[:, :, slice_idx], cmap="gray")
plt.title("Brain Mask")
plt.subplot(1, 3, 3)
plt.imshow(brain_only[:, :, slice_idx], cmap="gray")
plt.title("Brain Extracted")
plt.savefig(f"{filename}.png")
