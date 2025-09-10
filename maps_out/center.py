import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import label, find_objects

def calculate_bounding_box_from_volume(volume, intensity_threshold=0.1):
    # Normalize the volume
    volume_normalized = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    # Apply intensity threshold
    binary_mask = volume_normalized > intensity_threshold

    # Label connected components
    labeled_array, num_features = label(binary_mask)

    # Find the largest connected component
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # Exclude background
    largest_component = np.argmax(component_sizes)

    # Create a mask for the largest component
    brain_mask = labeled_array == largest_component

    # Find the bounding box of the largest component
    slices = find_objects(brain_mask.astype(int))[0]
    min_indices = [s.start for s in slices]
    max_indices = [s.stop - 1 for s in slices]

    return min_indices, max_indices

def crop_brain_volumes(brain_data):
    

        # Calculate bounding box from the brain volume
    min_indices, max_indices = calculate_bounding_box_from_volume(brain_data)

        # Crop the volume
    cropped_brain = brain_data[min_indices[0]:max_indices[0] + 1,
                                   min_indices[1]:max_indices[1] + 1,
                                   min_indices[2]:max_indices[2] + 1]
    return cropped_brain


# -------------------------------
# Input and output paths
# -------------------------------
in_path = "attention_3d_mapped_backprop.nii.gz"
out_path = "attention_3d_mapped_backprop_cropped_centered_resized.nii.gz"

# -------------------------------
# Load the NIfTI image
# -------------------------------
img = nib.load(in_path)
data = img.get_fdata()
affine = img.affine.copy()
header = img.header.copy()

data = crop_brain_volumes(data)

# -------------------------------
# 2. Resize in-plane slices to 160x160
# -------------------------------
# Assume sagittal slices → (X, Y, Z) where X = slice direction
target_size = (160, 160)
orig_y, orig_z = data.shape[1], data.shape[2]

zoom_factors = (
    1,  # keep sagittal slice count
    target_size[0] / orig_y,
    target_size[1] / orig_z
)

resized_data = zoom(data, zoom_factors, order=1)  # linear interpolation

# -------------------------------
# 3. Center affine so midpoint is at (0,0,0)
# -------------------------------
shape = np.array(resized_data.shape)

# Get voxel sizes (from affine if valid, otherwise assume 1mm)
if np.allclose(affine, np.eye(4)):
    voxel_sizes = np.ones(3)
else:
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

center_shift = -(shape[:3] / 2) * voxel_sizes

new_affine = affine.copy()
new_affine[:3, 3] = center_shift

# -------------------------------
# 4. Save new image
# -------------------------------
new_img = nib.Nifti1Image(resized_data, new_affine, header=header)
nib.save(new_img, out_path)

print(f"Cropped, centered & resized image saved to: {out_path}")
