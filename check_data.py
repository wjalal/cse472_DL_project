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
from dataset_cls import ADNIDataset, ADNIDatasetViT
from torch.utils.data import DataLoader, Dataset
import gc

model_name = "vit-small"

def set_random_seed(seed=69420):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

universal_seed = 69420

set_random_seed(universal_seed)


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


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the CSV file into a pandas DataFrame
csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
df_adni = pd.read_csv(csv_path)
# df = df.sample(n=1000, random_state=69420)
# Add a new column 'filepath' with the constructed file paths
df_adni['filepath'] = df_adni.apply(
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1
)
df_adni = df_adni.sort_values(by='Age', ascending=True).reset_index(drop=True)
df_adni = df_adni.head(900)
# df_adni=df_adni.sample(n=400)

# Load independent dataset metadata
metadata_path = "ixi_storage/ixi_brainrotnet_metadata.csv"
df_ixi = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_ixi['filepath'] = df_ixi.apply(
    lambda row: f"ixi_storage/IXI_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "abide_storage/abide_brainrotnet_metadata.csv"
df_abide = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_abide['filepath'] = df_abide.apply(
    lambda row: f"abide_storage/ABIDEII_bias_corrected/{row['ImageID'][7:]}.stripped.N4.nii.gz",
    axis=1
)
df_abide = df_abide.sort_values(by='Age', ascending=False).reset_index(drop=True)
df_abide = df_abide.head(750)
# df_abide=df_abide.sample(n=200)

metadata_path = "dlbs_storage/dlbs_brainrotnet_metadata.csv"
df_dlbs = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_dlbs['filepath'] = df_dlbs.apply(
    lambda row: f"dlbs_storage/DLBS_bias_corrected/{row['ImageID'][4:]}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "cobre_storage/cobre_brainrotnet_metadata.csv"
df_cobre = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_cobre['filepath'] = df_cobre.apply(
    lambda row: f"cobre_storage/COBRE_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "fcon1000_storage/fcon1000_brainrotnet_metadata.csv"
df_fcon = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_fcon['filepath'] = df_fcon.apply(
    lambda row: f"fcon1000_storage/fcon1000_bias_corrected/{row['ImageID'][8:]}.stripped.N4.nii.gz",
    axis=1
)
df_fcon = df_fcon.dropna()
# df_fcon = df_fcon.sort_values(by='Age', ascending=False).reset_index(drop=True).head(300)
# df_fcon = df_fcon.sample(n=300)

metadata_path = "sald_storage/sald_brainrotnet_metadata.csv"
df_sald = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_sald['filepath'] = df_sald.apply(
    lambda row: f"sald_storage/SALD_bias_corrected/sub-{row['ImageID'][4:]}.stripped.N4.nii.gz",
    axis=1
)
# df_sald = df_sald.sort_values(by='Age', ascending=False).reset_index(drop=True).head(300)
# df_sald = df_sald.sample(n=300)

metadata_path = "corr_storage/corr_brainrotnet_metadata.csv"
df_corr = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_corr['filepath'] = df_corr.apply(
    lambda row: f"corr_storage/CORR_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
    axis=1
)
df_corr = df_corr.sort_values(by='Age', ascending=True).reset_index(drop=True)
# df_corr = df_corr.head(300)
# df_corr = df_corr.sample(n=200)


metadata_path = "oasis1_storage/oasis1_brainrotnet_metadata.csv"
df_oas1 = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_oas1['filepath'] = df_oas1.apply(
    lambda row: f"oasis1_storage/oasis_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)
df_oas1 = df_oas1.sort_values(by='Age', ascending=False)
df_oas1 = df_oas1.reset_index(drop=True).head(300)

metadata_path = "camcan_storage/camcan_brainrotnet_metadata.csv"
df_camcan = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_camcan['filepath'] = df_camcan.apply(
    lambda row: f"camcan_storage/CamCAN_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "nimh_storage/nimh_mprage_brainrotnet_metadata.csv"
df_nimh = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_nimh['filepath'] = df_nimh.apply(
    lambda row: f"nimh_storage/nimh_bias_corrected/{row['ImageID'][5:]}_ses-01_acq-MPRAGE_rec-SCIC_T1w.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "bold_storage/bold_brainrotnet_metadata.csv"
df_bold = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_bold['filepath'] = df_bold.apply(
    lambda row: f"bold_storage/bold_bias_corrected/{row['ImageID'][5:]}_T1w.stripped.N4.nii.gz",
    axis=1
)

df = pd.concat ([
                 df_adni[['ImageID', 'Sex', 'Age', 'filepath']], 
                 df_ixi[['ImageID', 'Sex', 'Age', 'filepath']], 
                 df_abide[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_dlbs[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_cobre[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_fcon[['ImageID', 'Sex', 'Age', 'filepath']],
                #  df_sald[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_corr[['ImageID', 'Sex', 'Age', 'filepath']], 
                 df_oas1[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_camcan[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_nimh[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_bold[['ImageID', 'Sex', 'Age', 'filepath']]
                 ], ignore_index=True)
print (df)

# Ensure 'Age' is an integer
df['Age_Group'] = df['Age'].astype(int).apply(lambda x: f"{x:03d}"[:-1] + "0")
df['Age_Group'] = df['Age_Group'] + df['Sex']
print (df['Age_Group'].unique())
# Prepare dataset and dataloaders
sex_encoded = df['Sex'].apply(lambda x: 0 if x == 'M' else 1).tolist()
age_list = df['Age'].tolist()
filepath_list = df['filepath'].tolist()
label_list = df['Age_Group'].tolist()

df_test = pd.concat ([
        df_dlbs[['ImageID', 'Sex', 'Age', 'filepath']],

], ignore_index=True)

df_test['Age_Group'] = df_test['Age'].astype(int).apply(lambda x: f"{x:03d}"[:-1] + "0")
df_test['Age_Group'] = df_test['Age_Group'] + df_test['Sex']
print (df_test['Age_Group'].unique())
# Prepare test dataset
sex_encoded_test = df_test['Sex'].apply(lambda x: 0 if x == 'M' else 1).tolist()
age_list_test = df_test['Age'].tolist()
filepath_list_test = df_test['filepath'].tolist()
label_list_test = df_test['Age_Group'].tolist()


# Get unique labels and create a mapping
unique_labels = sorted(set(label_list))  # Ensure consistent ordering
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}  # Reverse mapping for decoding

# Convert labels to integers
numeric_labels = [label_to_idx[label] for label in label_list]
label_list = numeric_labels

# Convert test labels to integers
unique_labels_test = sorted(set(label_list_test))  # Ensure consistent ordering
label_to_idx_test = {label: idx for idx, label in enumerate(unique_labels_test)}
idx_to_label_test = {idx: label for label, idx in enumerate(unique_labels_test)}  # Reverse mapping for decoding
# Convert test labels to integers
numeric_labels_test = [label_to_idx_test[label] for label in label_list_test]
label_list_test = numeric_labels_test

roi = 160

# Transformation pipeline for ViT
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Convert to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize for ViT
])


# Instantiate Dataset
vit_dataset = ADNIDatasetViT(filepath_list, label_list)

# Split Dataset
train_size = int(0.8 * len(vit_dataset))
val_size = len(vit_dataset) - train_size
generator = torch.Generator().manual_seed(universal_seed)
vit_train_dataset, vit_val_dataset = torch.utils.data.random_split(vit_dataset, [train_size, val_size], generator=generator)

vit_dataset_test = ADNIDatasetViT(filepath_list_test, label_list_test)

train_ind = vit_train_dataset.indices
val_ind = vit_val_dataset.indices

train_ages = [age_list[i] for i in train_ind]
val_ages = [age_list[i] for i in val_ind]

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)  # Only share x-axis now

# Train histogram
axs[0].hist(train_ages, bins=30, color='skyblue', edgecolor='black')
axs[0].set_title(f"(a) Distribution of {len(train_ages)} training data")
axs[0].set_xlabel("Age")
axs[0].set_ylabel("Frequency")

# Validation histogram
axs[1].hist(val_ages, bins=30, color='skyblue', edgecolor='black')
axs[1].set_title(f"(b) Distribution of {len(val_ages)} validation data")
axs[1].set_xlabel("Age")
axs[1].set_ylabel("Frequency")

# Align layout nicely
plt.tight_layout()
plt.show()
