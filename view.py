from deepbrain import Extractor
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
# Load the NIfTI image
filename = "output_brain2.N4.nii"
nii_img = nib.load(filename)
data = nii_img.get_fdata()

# Visualize the results
slice_idx = data.shape[2] // 2
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(data[:, :, slice_idx], cmap="gray")
plt.title("Original Image")
plt.savefig(f"{filename}.png")
