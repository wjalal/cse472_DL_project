import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.ndimage import zoom 

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



# Load metadata
csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
df = pd.read_csv(csv_path)

# Append full file paths for .nii.gz images
data_dir = "roi"

def get_image_path(row):
    image_title = f"I{row['ImageID']}_{row['SubjectID']}.stripped.N4_cropped.nii.gz"
    return os.path.join(data_dir, image_title)

df['filepath'] = df.apply(get_image_path, axis=1)

# Drop rows where the filepath doesn't exist
df = df[df['filepath'].apply(os.path.exists)]

print(df.head())  # Verify file paths and Age column


def load_and_preprocess(filepath, patch_size=(7, 7)):
    """
    Load the .nii.gz file, resample it, extract three views, and divide them into fixed-size patches.
    The number of patches is dynamically calculated based on the patch size and image dimensions.
    """
    import nibabel as nib
    import torch

    # Load the .nii.gz image
    image = nib.load(filepath).get_fdata()
    image = resample_nifti(image, [91, 109, 91])
    
    # Extract the three views
    axial = image[:, :, image.shape[2] // 2]       # Axial view
    coronal = image[:, image.shape[1] // 2, :]     # Coronal view
    sagittal = image[image.shape[0] // 2, :, :]    # Sagittal view

    def extract_patches(view, patch_size):
        """
        Extract fixed-size patches from a 2D view and pad to handle dimensions not divisible by patch size.
        Dynamically calculate the number of patches.
        """
        h, w = view.shape
        ph, pw = patch_size

        # Calculate the number of patches in height and width
        num_patches_h = (h + ph - 1) // ph  # Ceiling division for height
        num_patches_w = (w + pw - 1) // pw  # Ceiling division for width
        num_patches = num_patches_h * num_patches_w

        # Pad the image to ensure dimensions are divisible by the patch size
        pad_h = (num_patches_h * ph - h)
        pad_w = (num_patches_w * pw - w)
        view = torch.nn.functional.pad(torch.tensor(view, dtype=torch.float32), (0, pad_w, 0, pad_h), value=0)

        # Extract patches
        patches = []
        for i in range(0, view.shape[0], ph):
            for j in range(0, view.shape[1], pw):
                patch = view[i:i + ph, j:j + pw]
                patches.append(patch.flatten())

        # Convert to tensor
        patches = torch.stack(patches)
        return patches, num_patches

    # Extract patches and calculate dynamic num_patches
    axial_patches, num_patches_axial = extract_patches(axial, patch_size)
    coronal_patches, num_patches_coronal = extract_patches(coronal, patch_size)
    sagittal_patches, num_patches_sagittal = extract_patches(sagittal, patch_size)

    # Verify consistency
    assert num_patches_axial == num_patches_coronal == num_patches_sagittal, \
        "Mismatch in the number of patches across views."

    return axial_patches, coronal_patches, sagittal_patches


def calculate_num_patches(image_size, patch_size):
    """
    Calculate the total number of patches for a given image size and patch size.

    Args:
        image_size (tuple): Dimensions of the image (height, width).
        patch_size (tuple): Dimensions of each patch (height, width).

    Returns:
        int: Total number of patches.
    """
    h, w = image_size
    ph, pw = patch_size

    # Calculate the number of patches along each dimension
    num_patches_h = (h + ph - 1) // ph  # Ceiling division for height
    num_patches_w = (w + pw - 1) // pw  # Ceiling division for width

    # Total patches
    return num_patches_h * num_patches_w

num_patches = calculate_num_patches((91, 109), (7, 7))



class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(x, x, x)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x

class TriameseViT(nn.Module):
    def __init__(self, num_patches, embed_dim=768, num_heads=12, mlp_dim=512, depth=10):
        super().__init__()
        self.patch_embedding = nn.Linear(256, embed_dim)  # Flattened patch size 16x16 -> 256
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # Three transformer encoders
        self.transformers = nn.ModuleList([
            nn.Sequential(*[TransformerEncoder(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
            for _ in range(3)
        ])

        # Three MLP heads with hidden layer size 3072
        self.mlp_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 3072),
                nn.ReLU(),
                nn.Linear(3072, 1)
            )
            for _ in range(3)
        ])

        # Final FNN layers: 1024 -> 512 -> 256 -> 128 -> 3 -> 1
        self.triamese_fnn = nn.Sequential(
            nn.Linear(3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, axial, coronal, sagittal):
        # Patch embedding + positional encoding
        views = [axial, coronal, sagittal]
        outputs = []

        for i, view in enumerate(views):
            x = self.patch_embedding(view) + self.pos_embedding
            x = self.transformers[i](x)
            x = x.mean(dim=1)  # Global average pooling
            outputs.append(self.mlp_heads[i](x))

        combined = torch.cat(outputs, dim=1)
        final_output = self.triamese_fnn(combined)
        return final_output
    

# Dataset Class
class MRIDataset(Dataset):
    def __init__(self, df):
        self.filepaths = df['filepath'].values
        self.ages = df['Age'].values

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        axial, coronal, sagittal = load_and_preprocess(filepath)
        label = torch.tensor(self.ages[idx], dtype=torch.float32)
        return axial, coronal, sagittal, label

# Split data into train and validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Create datasets
train_dataset = MRIDataset(train_df)
val_dataset = MRIDataset(val_df)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")


model = TriameseViT(num_patches=num_patches)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6, betas=(0.9, 0.999))
criterion = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training and Validation Loop
for epoch in range(20):
    # Training Phase
    model.train()
    train_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", unit="batch")
    
    for axial, coronal, sagittal, labels in train_loader_tqdm:
        axial, coronal, sagittal, labels = axial.to(device), coronal.to(device), sagittal.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(axial, coronal, sagittal)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        # Update tqdm with current loss
        train_loader_tqdm.set_postfix(loss=loss.item())
    
    # Validation Phase
    model.eval()
    val_loss = 0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]", unit="batch")
    
    with torch.no_grad():
        for axial, coronal, sagittal, labels in val_loader_tqdm:
            axial, coronal, sagittal, labels = axial.to(device), coronal.to(device), sagittal.to(device), labels.to(device)
            outputs = model(axial, coronal, sagittal)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Update tqdm with current loss
            val_loader_tqdm.set_postfix(loss=loss.item())
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")