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
from scipy.ndimage import zoom
import SimpleITK as sitk
import gzip
import shutil
# nib.openers.Opener.default_compresslevel = 9

def set_random_seed(seed=69420):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(69420)

import gzip
import os

def resample_nifti(img_data, target_slices = [64, 64, 32]):
    # Determine the current number of slices along the z-axis (3rd dimension)
    current_slices =  [img_data.shape[0], img_data.shape[1],  img_data.shape[2]]
    # Calculate the zoom factor for resampling (only along the z-axis)
    zoom_factor = [a / b for a, b in zip(target_slices, current_slices)]

    # Resample the image data along the z-axis
    resampled_data = zoom(img_data, (zoom_factor[0], zoom_factor[1], zoom_factor[2]), order=3)  # order=3 for cubic interpolation
    # Ensure that the resampled data has the target number of slices
    # print (resampled_data.shape)
    # resampled_data = resampled_data[:target_slices,:,:]
    # print (resampled_data.shape)
    return resampled_data

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV file into a pandas DataFrame
csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
df = pd.read_csv(csv_path).sample(n=120)
# df = df.sample(n=1000, random_state=69420)
print (df)
# Add a new column 'filepath' with the constructed file paths
df['filepath'] = df.apply(
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1
)


# Assuming `resample_volume_to_fixed_slices` and `transform` are predefined.
# Define constants
roi = 64  # Region of interest size (adjust as needed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple Linear Attention implementation
class LinearAttention(nn.Module):
    def __init__(self, embed_dim, downsample_factor=4):
        super(LinearAttention, self).__init__()
        self.embed_dim = embed_dim
        self.downsample_factor = downsample_factor
        self.kernel = nn.Linear(embed_dim, embed_dim)  # Simple kernel map for attention

    def forward(self, query, key, value):
        # Downsample the input to reduce the sequence length before attention
        batch, seq_len, _ = query.shape
        downsampled_len = seq_len // self.downsample_factor
        query_downsampled = query[:, :downsampled_len, :]
        key_downsampled = key[:, :downsampled_len, :]
        value_downsampled = value[:, :downsampled_len, :]

        # Apply kernel transformation (feature map)
        query_transformed = self.kernel(query_downsampled)
        key_transformed = self.kernel(key_downsampled)

        # Compute attention without the full matrix multiplication
        attention_scores = torch.bmm(query_transformed, key_transformed.transpose(1, 2))  # Batch matrix multiplication
        attention_scores = torch.softmax(attention_scores, dim=-1)  # Softmax for normalized attention scores
        
        # Apply attention scores to the values
        output_downsampled = torch.bmm(attention_scores, value_downsampled)  # Weighted sum of values
        
        # Restore the original sequence length
        # Option 1: Use nearest neighbor interpolation to upsample the output
        output_restored = torch.nn.functional.interpolate(output_downsampled, size=(seq_len,), mode='nearest')
        return output_restored

class BrainImageModel(nn.Module):
    def __init__(self):
        super(BrainImageModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        
        # Replaced Multihead Attention with Linear Attention
        self.linear_attention = LinearAttention(embed_dim=64)

        self.concat = lambda x, y: torch.cat((x, y), dim=1)
        self.batch_norm1 = nn.BatchNorm3d(128)
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm3d(128)
        self.elu2 = nn.ELU()
        self.conv4 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm3d(128)
        self.add = lambda x, y: x + y
        self.elu3 = nn.ELU()
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 16 * 8 * 128, 128)
        self.elu4 = nn.ELU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(129, 1)

    def forward(self, x, gender):
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x1))
        x2 = x2.permute(0, 2, 3, 4, 1)  # Rearrange for attention: (batch, depth, height, width, channels)
        
        batch, depth, height, width, channels = x2.shape
        x2 = x2.reshape(batch, depth * height * width, channels)  # Flatten spatial dimensions for attention
        print(x2.shape)

        # Use Linear Attention instead of Multihead Attention
        x2 = self.linear_attention(x2, x2, x2)  # Apply linear attention (query, key, value)
        print(f"After attention {x2.shape}")
        x2 = x2.reshape(batch, depth, height, width, channels).permute(0, 4, 1, 2, 3)  # Reshape back
        x = self.concat(x1, x2)  # Concatenate outputs from conv2 and attention
        x = self.batch_norm1(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = torch.relu(self.conv3(x))
        x = self.batch_norm2(x)
        x = self.elu2(x)
        x_residual = torch.relu(self.conv4(x))
        x_residual = torch.relu(self.conv5(x_residual))
        x_residual = self.batch_norm3(x_residual)
        x = self.add(x, x_residual)
        x = self.elu3(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.elu4(x)
        x = self.dropout(x)
        x = torch.cat((x, gender), dim=1)  # Concatenate gender input
        x = self.fc2(x)
        return x
    
model = BrainImageModel().to(device)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataset_cls import ADNIDatasetLite

# Prepare dataset and dataloaders
sex_encoded = df['Sex'].apply(lambda x: 0 if x == 'M' else 1).tolist()
age_list = df['Age'].tolist()

# Print feature shape for debugging
image_list = []

# Process images
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    filepath = row['filepath']
    image_title = f"{row['ImageID'][4:]}_{row['SubjectID']}"
    resampled_path = f"adni_storage/ADNI_bc_ro/{image_title}_ro.nii.gz"
    # print (resampled_path)
    if os.path.exists(filepath) and os.path.exists(resampled_path):
        # print ("loading")
        nii_img = nib.load(resampled_path)
        data = nii_img.get_fdata()
        image_list.append(data)
    else:
        # print ("loading")
        nii_img = nib.load(filepath)
        # print ("loaded")

        # Get current orientation and reorient to RAS
        orig_ornt = io_orientation(nii_img.affine)
        ras_ornt = axcodes2ornt(("R", "A", "S"))
        ornt_trans = ornt_transform(orig_ornt, ras_ornt)

        data = nii_img.get_fdata()
        data = nib.orientations.apply_orientation(data, ornt_trans)

        # Resample the volume to 160 slices
        data = resample_nifti(data, target_slices=[64,64,32])
        image_list.append(data)
        # Save the reoriented and resampled image
        nib.save(nib.Nifti1Image(data, nii_img.affine), resampled_path)


# Create Dataset and DataLoader
dataset = ADNIDatasetLite(image_list, sex_encoded, age_list)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Store the indices of the validation dataset
val_indices = val_dataset.indices
train_indices = train_dataset.indices

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print(f"Train loader size: {len(train_loader)}")
print(f"Validation loader size: {len(val_loader)}")

# Define the loss criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0007)

# Training and validation loop
num_epochs = 100

for epoch in range(1, num_epochs + 1):
    model.train()  # Set the model to training mode
    train_loss = 0.0

    # for images, sex, ages in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training"):
    for images, sex, ages in train_loader:
        images, sex, ages = images.to(device), sex.to(device), ages.to(device)
        images = images.unsqueeze(1)  # Add a channel dimension if missing

        # Forward pass
        outputs = model(images, sex)
        loss = criterion(outputs.squeeze(), ages.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for images, sex, ages in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - Validation"):
            images, sex, ages = images.to(device), sex.to(device), ages.to(device)

            # Forward pass
            outputs = model(images, sex)
            loss = criterion(outputs.squeeze(), ages.float())

            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)

    # Print epoch losses
    print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
