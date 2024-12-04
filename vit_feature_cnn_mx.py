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
print (df)
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

        # Define convolutional and pooling layers
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(10, 60), stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = nn.Conv2d(1, 1, kernel_size=(5, 15), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv3 = nn.Conv2d(1, 1, kernel_size=(2, 6), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.flatten = nn.Flatten()

        # Fully connected layers (fc1 dimensions are calculated dynamically)
        self.fc1 = None  # Placeholder to be initialized dynamically
        self.fc1_bn = None  # Placeholder for batch normalization after fc1
        self.fc2 = nn.Linear(512, 128)
        self.fc2_bn = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.1)  # Dropout with 10% probability
        self.fc3 = nn.Linear(129, 1)  # Adding 1 for the `Sex` input

        self.relu = nn.ReLU()
        self.initialize_fc1(input_shape)

    def initialize_fc1(self, input_shape):
        # Create a sample input to pass through the convolutional layers
        sample_input = torch.zeros(1, *input_shape)
        x = self.conv1(sample_input)
        x = self.relu(x)
        x = self.pool1(x)  # Apply pooling
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)  # Apply pooling
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)  # Apply pooling
        flattened_size = x.numel()  # Total number of elements after flattening
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc1_bn = nn.LayerNorm(512)  # Initialize batch normalization for fc1

    def forward(self, x, sex):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)  # Apply pooling
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)  # Apply pooling
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)  # Apply pooling
        x = self.flatten(x)

        if self.fc1 is None:
            raise ValueError("fc1 layer has not been initialized. Call `initialize_fc1` with the input shape.")

        x = self.fc1(x)
        x = self.fc1_bn(x)  # Apply batch normalization
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout

        x = self.fc2(x)
        x = self.fc2_bn(x)  # Apply batch normalization
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout

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

# Store the indices of the validation dataset
val_indices = val_dataset.indices
train_indices = train_dataset.indices

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


# Tracking outputs for validation samples
val_outputs = {}
train_outputs = {}

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AgePredictionCNN(features_list[0].shape).to(device)
criterion = nn.L1Loss()  # MAE Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_loss = np.inf  # Initialize the best loss to infinity
start_epoch = 0

load_saved = "none" # "last, "best"
if load_saved != "none":
    # Load the checkpoint
    with open(f"model_dumps/{sys.argv[0][:-3]}_{load_saved}_model_with_metadata.pkl", "rb") as f:
        checkpoint = pickle.load(f)

    # Restore model and optimizer state
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    # Restore RNG states
    torch.set_rng_state(checkpoint["t_rng_st"])
    np.random.set_state(checkpoint["n_rng_st"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_st"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_st"])

    # Retrieve metadata
    start_epoch = checkpoint["epoch"] + 1
    loaded_loss = checkpoint["loss"]

    print(f"Loaded model from epoch {start_epoch} with best validation loss: {loaded_loss:.4f}")

predicted_ages = None
# Training loop
epochs = 200
for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = 0.0
    predicted_ages = []
    for idx, (features, sex, age) in enumerate(train_loader):
        features = features.unsqueeze(1).to(device)  # Add channel dimension
        sex = sex.to(device)
        age = age.to(device)
        optimizer.zero_grad()
        outputs = model(features, sex)
        train_outputs[train_indices[idx]] = outputs.item()
        # print (outputs)
        loss = criterion(outputs.squeeze(), age)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    # predicted_ages = []
    with torch.no_grad():
        for idx, (features, sex, age) in enumerate(val_loader):
            features = features.unsqueeze(1).to(device)
            sex = sex.to(device)
            age = age.to(device)

            outputs = model(features, sex)
            # predicted_ages.append(outputs.item())
            loss = criterion(outputs.squeeze(), age)

            val_loss += loss.item()

             # Save the predicted age for the current validation sample
            val_outputs[val_indices[idx]] = outputs.item()

    # print (predicted_ages)
    val_loss /= len(val_loader)



    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")
    print(f"Saving last model...")
    checkpoint = {  # Add or update keys in the checkpoint
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "loss": val_loss,
        "t_rng_st": torch.get_rng_state(),
        "n_rng_st": np.random.get_state(),
        "cuda_rng_st": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    with open("model_dumps/last_model_with_metadata.pkl", "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Last model saved...")


    # Check if validation loss improved and save the model with metadata
    if val_loss < best_loss:
        best_loss = val_loss
        print(f"Validation loss improved to {best_loss:.4f}. Saving model...")
        checkpoint = {  # Add or update keys in the checkpoint
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "loss": best_loss,
            "t_rng_st": torch.get_rng_state(),
            "n_rng_st": np.random.get_state(),
            "cuda_rng_st": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        with open("model_dumps/best_model_with_metadata.pkl", "wb") as f:
            pickle.dump(checkpoint, f)

    max_index = max(train_outputs.keys())
    # Create a DataFrame with NaN for all indices initially
    df_trn = pd.DataFrame(index=range(max_index + 1), columns=["Predicted_Age"])
    # Assign the values to their respective indices
    for index, value in train_outputs.items():
        df_trn.loc[index, "Predicted_Age"] = value
    print (df_trn)

    df2 = df.copy()
    df2['Predicted_Age'] = df_trn['Predicted_Age']
    train_df = df2.loc[train_outputs.keys()]
    print (train_df)
    train_df.to_csv("predicted_ages_train.csv")

    max_index = max(val_outputs.keys())
    # Create a DataFrame with NaN for all indices initially
    df_pred = pd.DataFrame(index=range(max_index + 1), columns=["Predicted_Age"])
    # Assign the values to their respective indices
    for index, value in val_outputs.items():
        df_pred.loc[index, "Predicted_Age"] = value
    print (df_pred)

    df1 = df.copy()
    df1['Predicted_Age'] = df_pred['Predicted_Age']
    test_df = df1.loc[val_outputs.keys()]
    print (test_df)
    test_df.to_csv("predicted_ages_val.csv")


    # Check that the predictions have been added to the DataFrame
    # Plot Age vs. Predicted Age
    # plt.figure(figsize=(8, 6))
    # plt.scatter(train_df['Age'], train_df['Predicted_Age'], color='blue', label='Predicted vs Actual')
    # # plt.plot(test_df['Age'], test_df['Age'], color='red', linestyle='--', label='Perfect Prediction')  # Optional: Line of perfect prediction
    # plt.xlabel('Age')
    # plt.ylabel('Predicted Age')
    # plt.title('Age vs Predicted Age')
    # plt.legend()
    # plt.grid(True)
    # plt.show()