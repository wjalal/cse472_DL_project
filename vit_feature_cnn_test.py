import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from tqdm import tqdm
from torchvision import transforms
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
from transformers import ViTFeatureExtractor, ViTModel
import pickle
import sys
import SimpleITK as sitk
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
from dataset_cls import ADNIDataset
import matplotlib.pyplot as plt

# Function to resample the volume to 160 slices
def resample_volume_to_fixed_slices(data, affine, target_slices=160):
    # Convert Numpy array and affine to SimpleITK image
    sitk_img = sitk.GetImageFromArray(data)
    sitk_img.SetSpacing([affine[0, 0], affine[1, 1], affine[2, 2]])

    original_size = sitk_img.GetSize()  # (width, height, depth)
    original_spacing = sitk_img.GetSpacing()  # (spacing_x, spacing_y, spacing_z)

    # Calculate new spacing to achieve the target number of slices
    new_spacing = list(original_spacing)
    new_spacing[2] = (original_spacing[2] * original_size[2]) / target_slices

    # Define new size
    new_size = [original_size[0], original_size[1], target_slices]

    # Resample the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_img = resampler.Execute(sitk_img)

    return sitk.GetArrayFromImage(resampled_img)  # Return the resampled image as a numpy array


def resample_nifti(img_data, target_slices = 160):
    # Determine the current number of slices along the z-axis (3rd dimension)
    current_slices = img_data.shape[0]
    # Calculate the zoom factor for resampling (only along the z-axis)
    zoom_factor = target_slices / current_slices
    # Resample the image data along the z-axis
    resampled_data = zoom(img_data, (zoom_factor, 1, 1), order=3)  # order=3 for cubic interpolation
    # Ensure that the resampled data has the target number of slices
    # print (resampled_data.shape)
    # resampled_data = resampled_data[:target_slices,:,:]
    # print (resampled_data.shape)
    return resampled_data

###########################################
# THIS IS WHERE YOU CHOOSE A MODEL TO TEST 
###########################################

import importlib

# Assuming sys.argv[1] is the module name
module_name = sys.argv[1]  # Example: "my_model"
class_name = "AgePredictionCNN"  # The class you want to import

try:
    # Dynamically import the module
    module = importlib.import_module(module_name)
    
    # Dynamically get the class
    AgePredictionCNN = getattr(module, class_name)
    
    print(f"Successfully imported {class_name} from {module_name}.")

except ImportError:
    print(f"Module {module_name} could not be imported.")
except AttributeError:
    print(f"{class_name} does not exist in {module_name}.")

##############################
# MODEL IMPORTED DYNAMICALLY
##############################


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


metadata_path = "corr_storage/corr_brainrotnet_metadata.csv"
df_corr = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_corr['filepath'] = df_corr.apply(
    lambda row: f"corr_storage/CORR_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
    axis=1
)
df_corr = df_corr.sort_values(by='Age', ascending=True).reset_index(drop=True)
df_test = df_corr

# metadata_path = "oasis1_storage/oasis1_brainrotnet_metadata.csv"
# df_oas1 = pd.read_csv(metadata_path)
# # Update filepaths for the independent dataset
# df_oas1['filepath'] = df_oas1.apply(
#     lambda row: f"oasis1_storage/oasis_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
#     axis=1
# )
# df_test = df_oas1

# Load ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
model.to(device)
model.eval()

# Transform for images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# Function to process test dataset
features_list = []
labels_list = []

for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Processing test images"):
    filepath = row['filepath']    
    image_title = f"{row['ImageID'][5:]}"
        # Check if the feature file already exists
    # feature_file_path = f"abide_storage/ABIDEII_features/{image_title}_features.npy"
    feature_file_path = f"corr_storage/CORR_features/{image_title}_features.npy"
    if os.path.exists(feature_file_path):
        # If file exists, load the features from the file
        features = np.load(feature_file_path)


        # plt.figure(figsize=(7.68, 1.6), dpi=1)
        # plt.imshow(features, aspect='auto', cmap='gray')
        # plt.axis('off')  # Turn off axes
        # # Save the heatmap as an image
        # plt.savefig(f"corr_storage/CORR_features/featuremaps/{image_title}_fm.png", dpi=130, bbox_inches='tight', pad_inches=0)  # No padding
        # plt.close()  # Close the plot to free memory

        from PIL import Image
        # Normalize the array to 0-255 for grayscale image
        data_normalized = ((features - np.min(features)) / (np.max(features) - np.min(features)) * 255).astype(np.uint8)
        data_normalized = np.repeat(data_normalized, 4, axis=0)
        # Create an image from the array
        img = Image.fromarray(np.transpose(data_normalized), mode='L')  # 'L' mode for grayscale
        # Save the image
        img.save(f"corr_storage/CORR_features/featuremaps/{image_title}_fm.png")

        features_list.append(features)  # Flatten the features and add to the list
        labels_list.append(row['Age'])  # Add the corresponding age label
    else:
        if os.path.exists(filepath):
            try:
                # Load the NIfTI image
                nii_img = nib.load(filepath)

                # Get current orientation and reorient to RAS
                orig_ornt = io_orientation(nii_img.affine)
                ras_ornt = axcodes2ornt(("R", "A", "S"))
                ornt_trans = ornt_transform(orig_ornt, ras_ornt)

                data = nii_img.get_fdata()  # Load image data
                data = apply_orientation(data, ornt_trans)

                affine = nii_img.affine  # Affine transformation matrix

                # Resample the volume to 160 slices (if required)
                data = resample_nifti(data, target_slices=160)
                # Extract features for all slices

                # Visualize the results
                # slice_idx = data.shape[0] // 2
                # plt.figure(figsize=(12, 6))
                # plt.subplot(1, 3, 1)
                # plt.imshow(data[slice_idx, :, :], cmap="gray")
                # plt.title("Original Image")
                # plt.savefig(f"model_dumps/view_corr/{image_title}.png")


                features = []
                for slice_idx in range(data.shape[0]):
                    slice_data = data[slice_idx, :, :]
                    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
                    slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
                    # Extract features using ViT
                    with torch.no_grad():
                        outputs = model(slice_tensor)
                        slice_features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                        features.append(slice_features)
                # Save extracted features
                features = np.array(features)
                np.save(feature_file_path, features)
                features_list.append(features)
                labels_list.append(row['Age'])  # Assuming 'Age' is the target

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")

print ("data :" , features_list[0].shape)
sex_encoded = df_test['Sex'].apply(lambda x: 0 if x == 'M' else 1).tolist()
age_list = df_test['Age'].tolist()
dataset = ADNIDataset(features_list, sex_encoded, age_list)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load pre-trained model for age prediction
with open(f"model_dumps/mix/{sys.argv[1]}_best_model_with_metadata.pkl", "rb") as f:
    checkpoint = pickle.load(f)

age_model = AgePredictionCNN(features_list[0].shape).to(device)
age_model.load_state_dict(checkpoint["model_state"])
age_model.eval()
criterion = nn.L1Loss()  # MAE Loss

# Test the model
predicted_ages = []
test_loss = 0.0
with torch.no_grad():
    for features, sex, age in test_loader:
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(1).to(device)
        sex = sex.to(device)
        age = age.to(device)
        output = age_model(features, sex)  
        loss = criterion(output.squeeze(), age)
        test_loss += loss.item()
        predicted_ages.append(output.squeeze().cpu().numpy())

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")
# Save results
df_test['Predicted_Age'] = predicted_ages
# df_test.to_csv("model_dumps/abide_predicted_ages.csv", index=False)
# print("Predicted ages saved to 'model_dumps/abide_predicted_ages.csv'")

df_test.to_csv("model_dumps/mix/ixi_predicted_ages.csv", index=False)
print("Predicted ages saved to 'model_dumps/ixi_predicted_ages.csv'")
