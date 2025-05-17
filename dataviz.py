import os
import pandas as pd
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from tqdm import tqdm
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import sys
import SimpleITK as sitk
from scipy.ndimage import zoom
from dataset_cls import ADNIDatasetV2, ADNIDatasetViT
from torch.utils.data import DataLoader, Dataset
import gc

# def set_random_seed(seed=69420):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# universal_seed = 69420

# set_random_seed(universal_seed)

# # Function to resample the volume to 160 slices
# def resample_volume_to_fixed_slices(data, affine, target_slices=160):
#     # Convert Numpy array and affine to SimpleITK image
#     sitk_img = sitk.GetImageFromArray(data)
#     sitk_img.SetSpacing([affine[0, 0], affine[1, 1], affine[2, 2]])

#     original_size = sitk_img.GetSize()  # (width, height, depth)
#     original_spacing = sitk_img.GetSpacing()  # (spacing_x, spacing_y, spacing_z)

#     # Calculate new spacing to achieve the target number of slices
#     new_spacing = list(original_spacing)
#     new_spacing[2] = (original_spacing[2] * original_size[2]) / target_slices

#     # Define new size
#     new_size = [original_size[0], original_size[1], target_slices]

#     # Resample the image
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetOutputSpacing(new_spacing)
#     resampler.SetSize(new_size)
#     resampler.SetInterpolator(sitk.sitkLinear)
#     resampled_img = resampler.Execute(sitk_img)

#     return sitk.GetArrayFromImage(resampled_img)  # Return the resampled image as a numpy array


# def resample_nifti(img_data, target_slices = 160):
#     # Determine the current number of slices along the z-axis (3rd dimension)
#     current_slices = img_data.shape[0]
#     # Calculate the zoom factor for resampling (only along the z-axis)
#     zoom_factor = target_slices / current_slices
#     # Resample the image data along the z-axis
#     resampled_data = zoom(img_data, (zoom_factor, 1, 1), order=3)  # order=3 for cubic interpolation
#     # Ensure that the resampled data has the target number of slices
#     # print (resampled_data.shape)
#     # resampled_data = resampled_data[:target_slices,:,:]
#     # print (resampled_data.shape)
#     return resampled_data


# # Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the CSV file into a pandas DataFrame
# csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
# df_adni = pd.read_csv(csv_path)
# # df = df.sample(n=1000, random_state=69420)
# # Add a new column 'filepath' with the constructed file paths
# df_adni['filepath'] = df_adni.apply(
#     lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
#     axis=1
# )
# df_adni = df_adni.sort_values(by='Age', ascending=True).reset_index(drop=True).head(500)


# metadata_path = "dlbs_storage/dlbs_brainrotnet_metadata.csv"
# df_dlbs = pd.read_csv(metadata_path)
# # Update filepaths for the independent dataset
# df_dlbs['filepath'] = df_dlbs.apply(
#     lambda row: f"dlbs_storage/DLBS_bias_corrected/{row['ImageID'][4:]}.stripped.N4.nii.gz",
#     axis=1
# )


# metadata_path = "fcon1000_storage/fcon1000_brainrotnet_metadata.csv"
# df_fcon = pd.read_csv(metadata_path)
# # Update filepaths for the independent dataset
# df_fcon['filepath'] = df_fcon.apply(
#     lambda row: f"fcon1000_storage/fcon1000_bias_corrected/{row['ImageID'][8:]}.stripped.N4.nii.gz",
#     axis=1
# )
# df_fcon = df_fcon.dropna()

# metadata_path = "camcan_storage/camcan_brainrotnet_metadata.csv"
# df_camcan = pd.read_csv(metadata_path)
# # Update filepaths for the independent dataset
# df_camcan['filepath'] = df_camcan.apply(
#     lambda row: f"camcan_storage/CamCAN_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
#     axis=1
# )

# df = pd.concat([
#     df_adni[['ImageID', 'Sex', 'Age', 'filepath']], 
#     df_dlbs[['ImageID', 'Sex', 'Age', 'filepath']],
#     df_fcon[['ImageID', 'Sex', 'Age', 'filepath']],
#     df_camcan[['ImageID', 'Sex', 'Age', 'filepath']]
# ], ignore_index=True)

# class DataFrameDataset(Dataset):
#     def __init__(self, dataframe):
#         self.dataframe = dataframe

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         return self.dataframe.iloc[idx]

# wrapped_dataset = DataFrameDataset(df)

# generator = torch.Generator().manual_seed(universal_seed)

# train_size = int(0.8 * len(wrapped_dataset))
# val_size = len(wrapped_dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(wrapped_dataset, [train_size, val_size], generator=generator)

# df_train = pd.DataFrame([item for item in train_dataset])
# df_val = pd.DataFrame([item for item in val_dataset])

# df_train.to_csv("train_dataset.csv", index=False)
# df_val.to_csv("val_dataset.csv", index=False)

# Load datasets
# df_train = pd.read_csv("train_dataset.csv")
# df_val = pd.read_csv("val_dataset.csv")

#combine the datasets and save them in a csv

# df_combined = pd.concat([df_train, df_val], ignore_index=True)
# df_combined.to_csv("combined_dataset.csv", index=False)
# Load the combined dataset


# import pandas as pd
# import numpy as np

# df = pd.read_csv("combined_dataset.csv")

# # Initialize dictionaries to store age data for each dataset
# adni_ages = []
# dlbs_ages = []
# fcon_ages = []
# camcan_ages = []

# # Loop through the dataframe and classify rows based on file path
# for index, row in df.iterrows():
#     if 'adni_storage' in row['filepath']:
#         adni_ages.append(row['Age'])
#     elif 'dlbs_storage' in row['filepath']:
#         dlbs_ages.append(row['Age'])
#     elif 'fcon1000_storage' in row['filepath']:
#         fcon_ages.append(row['Age'])
#     elif 'camcan_storage' in row['filepath']:
#         camcan_ages.append(row['Age'])

# # TOTAL NUMBER OF IMAGES
# print(f"ADNI: {len(adni_ages)}")
# print(f"DLBS: {len(dlbs_ages)}")
# print(f"FCON: {len(fcon_ages)}")
# print(f"Cam-CAN: {len(camcan_ages)}")

# # Calculate mean and standard deviation for each dataset
# adni_mean, adni_std = np.mean(adni_ages), np.std(adni_ages)
# dlbs_mean, dlbs_std = np.mean(dlbs_ages), np.std(dlbs_ages)
# fcon_mean, fcon_std = np.mean(fcon_ages), np.std(fcon_ages)
# camcan_mean, camcan_std = np.mean(camcan_ages), np.std(camcan_ages)

# # Print the results
# print(f"ADNI: {adni_mean:.2f} ± {adni_std:.2f}")
# print(f"DLBS: {dlbs_mean:.2f} ± {dlbs_std:.2f}")
# print(f"FCON: {fcon_mean:.2f} ± {fcon_std:.2f}")
# print(f"Cam-CAN: {camcan_mean:.2f} ± {camcan_std:.2f}")

# # compute the range of ages for each dataset
# print("Age ranges:")
# print(f"ADNI: {min(adni_ages)} - {max(adni_ages)}")
# print(f"DLBS: {min(dlbs_ages)} - {max(dlbs_ages)}")
# print(f"FCON: {min(fcon_ages)} - {max(fcon_ages)}")
# print(f"Cam-CAN: {min(camcan_ages)} - {max(camcan_ages)}")

# # Load the datasets

# df_train = pd.read_csv("train_dataset.csv")
# df_val = pd.read_csv("val_dataset.csv")

# print(len(df_train))
# print(len(df_val))

# count_train_adni = df_train['filepath'].str.contains('adni_storage').sum()
# count_train_dlbs = df_train['filepath'].str.contains('dlbs_storage').sum()
# count_train_fcon = df_train['filepath'].str.contains('fcon1000_storage').sum()
# count_train_camcan = df_train['filepath'].str.contains('camcan_storage').sum()
# print(f"Train ADNI: {count_train_adni}")
# print(f"Train DLBS: {count_train_dlbs}")
# print(f"Train FCON: {count_train_fcon}")
# print(f"Train Cam-CAN: {count_train_camcan}")

# count_val_adni = df_val['filepath'].str.contains('adni_storage').sum()
# count_val_dlbs = df_val['filepath'].str.contains('dlbs_storage').sum()
# count_val_fcon = df_val['filepath'].str.contains('fcon1000_storage').sum()
# count_val_camcan = df_val['filepath'].str.contains('camcan_storage').sum()

# print(f"Val ADNI: {count_val_adni}")
# print(f"Val DLBS: {count_val_dlbs}")
# print(f"Val FCON: {count_val_fcon}")
# print(f"Val Cam-CAN: {count_val_camcan}")

