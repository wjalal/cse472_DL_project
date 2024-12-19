import os
import glob
import nibabel as nib
import numpy as np

def main():
    # Paths to dataset folders
    dataset_path = "ixi"
    output_folder = "ixiclroi"    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Retrieve file paths for brain volumes (supporting .nii.gz)
    brain_volume_list = sorted(glob.glob(os.path.join(dataset_path, '**/*.nii.gz'), recursive=True))

    # Crop and save volumes
    crop_and_save_brain_volumes(brain_volume_list, output_folder)

    print("Cropped brain volumes saved to:", output_folder)


def calculate_bounding_box_from_volume(volume):
    # Find indices of non-zero values
    non_zero_indices = np.argwhere(volume > 0)

    # Calculate min and max indices along each dimension
    min_indices = np.min(non_zero_indices, axis=0)
    max_indices = np.max(non_zero_indices, axis=0)

    # Convert indices to integers
    min_indices = min_indices.astype(int)
    max_indices = max_indices.astype(int)

    return min_indices, max_indices


def crop_and_save_brain_volumes(brain_volume_list, output_folder):
    for brain_path in brain_volume_list:
        # Load the brain volume
        brain_volume = nib.load(brain_path)
        brain_data = brain_volume.get_fdata()

        # Calculate bounding box from the brain volume
        min_indices, max_indices = calculate_bounding_box_from_volume(brain_data)

        # Crop the volume
        cropped_brain = brain_data[min_indices[0]:max_indices[0] + 1,
                                   min_indices[1]:max_indices[1] + 1,
                                   min_indices[2]:max_indices[2] + 1]

        # Save the cropped brain volume
        _, filename = os.path.split(brain_path)
        filename_without_extension = os.path.splitext(os.path.splitext(filename)[0])[0]  # Handles .nii.gz
        cropped_output_path = os.path.join(output_folder, filename_without_extension + "_cropped.nii.gz")

        nib.save(nib.Nifti1Image(cropped_brain, brain_volume.affine), cropped_output_path)

        print(f"Processed and saved: {filename}")


if __name__ == "__main__":
    main()
