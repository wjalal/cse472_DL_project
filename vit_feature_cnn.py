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

import SimpleITK as sitk

def set_random_seed(seed=69420):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(69420)

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

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV file into a pandas DataFrame
csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
df = pd.read_csv(csv_path)
# df = df.sample(n=1000, random_state=69420)
print (df.shape)
# Add a new column 'filepath' with the constructed file paths
df['filepath'] = df.apply(
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID']}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1
)

# Load pre-trained ViT model
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
model.to(device)  # Move the model to the GPU (if available)
model.eval()

# Update image transform for grayscale images to match ViT input requirements
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Convert to RGB directly
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# Directory to save processed images and features
os.makedirs("adni_storage/ADNI_features", exist_ok=True)

# Load pre-trained ViT model
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
model.to(device)  # Move the model to the GPU (if available)
model.eval()

# Update image transform for grayscale images to match ViT input requirements
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Convert to RGB directly
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# Directory to save processed images and features
os.makedirs("adni_storage/ADNI_features", exist_ok=True)

# To store features and labels
features_list = []
labels_list = []

# Process each row in the DataFrame
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    filepath = row['filepath']
    image_title = f"{row['ImageID']}_{row['SubjectID']}"

    # Check if the feature file already exists
    feature_file_path = f"adni_storage/ADNI_features/{image_title}_features.npy"
    if os.path.exists(feature_file_path):
        # If file exists, load the features from the file
        features = np.load(feature_file_path)
        features_list.append(features)  # Flatten the features and add to the list
        labels_list.append(row['Age'])  # Add the corresponding age label
    else:
        # print ("hiii")
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
                data = resample_volume_to_fixed_slices(data, affine, target_slices=160)

                # Extract features for all sagittal slices
                features = []
                for slice_idx in range(data.shape[0]):
                    slice_data = data[slice_idx, :, :]
                    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize

                    # Transform slice for ViT input
                    slice_tensor = transform(slice_data).unsqueeze(0).to(device)  # Add batch dimension and move to GPU

                    # Extract features using ViT
                    with torch.no_grad():
                        outputs = model(slice_tensor)
                        slice_features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Move output back to CPU
                        features.append(slice_features)

                # Save extracted features
                features = np.array(features)
                np.save(feature_file_path, features)
                features_list.append(features)
                labels_list.append(row['Age'])  # Target is 'Age'

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Custom Dataset
class ADNIDataset(Dataset):
    def __init__(self, features_list, sex_list, age_list):
        self.features = features_list
        self.sex = sex_list
        self.age = age_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.sex[idx], dtype=torch.float32),
            torch.tensor(self.age[idx], dtype=torch.float32),
        )

class AgePredictionCNN(nn.Module):
    def __init__(self, input_shape):
        super(AgePredictionCNN, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(10, 60), stride=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(5, 15), stride=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(2, 6), stride=1)

        self.flatten = nn.Flatten()

        # Fully connected layers (fc1 dimensions are calculated dynamically)
        self.fc1 = None  # Placeholder to be initialized dynamically
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(129, 1)  # Adding 1 for the `Sex` input

        self.relu = nn.ReLU()
        self.initialize_fc1(input_shape)

    def initialize_fc1(self, input_shape):
        # Create a sample input to pass through the convolutional layers
        sample_input = torch.zeros(1, *input_shape)
        x = self.conv1(sample_input)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        flattened_size = x.numel()  # Total number of elements after flattening
        self.fc1 = nn.Linear(flattened_size, 512)

    def forward(self, x, sex):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)

        if self.fc1 is None:
            raise ValueError("fc1 layer has not been initialized. Call `initialize_fc1` with the input shape.")

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        # Concatenate `Sex` input
        x = torch.cat((x, sex.unsqueeze(1)), dim=1)
        x = self.fc3(x)

        return x

# Prepare dataset and dataloaders
sex_encoded = df['Sex'].apply(lambda x: 0 if x == 'M' else 1).tolist()
age_list = df['Age'].tolist()

# print (features_list)
print (features_list[0].shape)

# Create Dataset and DataLoader
dataset = ADNIDataset(features_list, sex_encoded, age_list)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgePredictionCNN(features_list[0].shape).to(device)
criterion = nn.L1Loss()  # MAE Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

outputs = None
# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for features, sex, age in train_loader:
        # print (features.shape)
        # print (features)
        features = features.unsqueeze(1).to(device)  # Add channel dimension
        # print (features.shape)
        # print (features)
        sex = sex.to(device)
        age = age.to(device)

        optimizer.zero_grad()
        outputs = model(features, sex)
        loss = criterion(outputs.squeeze(), age)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, sex, age in val_loader:
            features = features.unsqueeze(1).to(device)
            sex = sex.to(device)
            age = age.to(device)

            outputs = model(features, sex)
            loss = criterion(outputs.squeeze(), age)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")


