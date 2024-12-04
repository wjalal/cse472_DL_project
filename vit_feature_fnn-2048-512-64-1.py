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
df = df.sample(n=1000, random_state=69420)
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
import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV file into a pandas DataFrame
csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
df = pd.read_csv(csv_path)
df = df.sample(n=1000, random_state=69420)  # Sample a subset of data
print(df.shape)

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
        features_list.append(features.flatten())  # Flatten the features and add to the list
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

                # Combine extracted features with 'Sex' value (0 for Male, 1 for Female)
                sex_value = 1 if row['Sex'] == 'F' else 0
                combined_features = np.concatenate([features.flatten(), [sex_value]])  # Flatten the 3D features and add sex value

                features_list.append(combined_features)
                labels_list.append(row['Age'])  # Target is 'Age'

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")

# Convert lists to numpy arrays
X = np.array(features_list)
print (X.shape)
y = np.array(labels_list)
print (y.shape)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the Feed-Forward Neural Network architecture with LayerNorm, Dropout, and He initialization
class FNN(nn.Module):
    def __init__(self, input_dim):
        super(FNN, self).__init__()
        
        # Layer 1: (n_input) x 2048
        self.fc1 = nn.Linear(input_dim, 2048)
        self.ln1 = nn.LayerNorm(2048)  # LayerNorm after the first layer
        self.dropout1 = nn.Dropout(0.1)  # Dropout after the first layer
        
        # Layer 2: 2048 x 512
        self.fc2 = nn.Linear(2048, 512)
        self.ln2 = nn.LayerNorm(512)  # LayerNorm after the second layer
        self.dropout2 = nn.Dropout(0.1)  # Dropout after the second layer
        
        # Layer 3: 512 x 64
        self.fc3 = nn.Linear(512, 64)
        self.ln3 = nn.LayerNorm(64)  # LayerNorm after the third layer
        self.dropout3 = nn.Dropout(0.1)  # Dropout after the third layer
        
        # Layer 4: 64 x 1 (output layer)
        self.fc4 = nn.Linear(64, 1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Layer 1: Fully Connected, LayerNorm, ReLU, Dropout
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        # Layer 2: Fully Connected, LayerNorm, ReLU, Dropout
        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        # Layer 3: Fully Connected, LayerNorm, ReLU, Dropout
        x = self.fc3(x)
        x = self.ln3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        # Layer 4: Output Layer (no activation for regression task)
        x = self.fc4(x)
        
        return x

    def _initialize_weights(self):
        # He Initialization (using normal distribution)
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='relu')

        # Initialize biases to 0
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fc3.bias)
        init.zeros_(self.fc4.bias)
# Initialize the model
model = FNN(input_dim=X_train.shape[1]).to(device)  # Input dimension is the number of features

# Loss function and optimizer
criterion = nn.L1Loss()  # MAE loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 150
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}] starting...")  # Print before each epoch starts

    model.train()
    optimizer.zero_grad()

    # Convert to torch tensors and move to device
    inputs = torch.tensor(X_train, dtype=torch.float32).to(device)
    targets = torch.tensor(y_train, dtype=torch.float32).to(device)

    # Forward pass
    outputs = model(inputs).squeeze()

    # Compute loss
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Evaluate on training data
    model.eval()
    with torch.no_grad():
        train_outputs = model(inputs).squeeze()
        train_loss = criterion(train_outputs, targets).item()

    # Evaluate on test data
    with torch.no_grad():
        inputs_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        targets_test = torch.tensor(y_test, dtype=torch.float32).to(device)
        test_outputs = model(inputs_test).squeeze()
        test_loss = criterion(test_outputs, targets_test).item()

    # Print train and test loss every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Final evaluation on test set
model.eval()
with torch.no_grad():
    inputs_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    targets_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    test_outputs = model(inputs_test).squeeze()
    mae = criterion(test_outputs, targets_test)
    print(f"Final Test MAE: {mae.item():.4f}")
