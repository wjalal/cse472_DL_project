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
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
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
    image_title = f"{row['ImageID'][4:]}_{row['SubjectID']}"

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
from dataset_cls import ADNIDataset

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

model = AgePredictionCNN(features_list[0].shape).to(device)
criterion = nn.L1Loss()  # MAE Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
# best_loss = np.inf  # Initialize the best loss to infinity
start_epoch = 0

with open(f"model_dumps/{sys.argv[1]}_best_model_with_metadata.pkl", "rb") as f:
    checkpoint = pickle.load(f)
best_loss = checkpoint["loss"]

load_saved = sys.argv[2] # "last, "best"
if load_saved != "none":
    # Load the checkpoint
    with open(f"model_dumps/{sys.argv[1]}_{load_saved}_model_with_metadata.pkl", "rb") as f:
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

    print(f"Loaded model from epoch {start_epoch} with validation loss {loaded_loss:.4f}, best loss {best_loss:.4f}")

predicted_ages = None
# Training loop
epochs = int(sys.argv[3])

# Initialize lists to track loss
filename = sys.argv[1] 
csv_file = f"model_dumps/{filename}.csv"

# Load existing epoch data if the file exists
if os.path.exists(csv_file):
    epoch_data = pd.read_csv(csv_file).to_dict(orient="records")
    print(f"Loaded existing epoch data from {csv_file}.")
else:
    epoch_data = []
    print("No existing epoch data found. Starting fresh.")


# Plot loss vs. epoch and save the figure
def update_loss_plot(epoch_data, filename):
    df = pd.DataFrame(epoch_data)
    df.to_csv(f"model_dumps/{filename}.csv", index=False)  # Save the data to CSV
    
    plt.figure(figsize=(8, 6))
    plt.plot(df['epoch'], df['train_loss'], label="Train Loss", marker="o")
    plt.plot(df['epoch'], df['val_loss'], label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"model_dumps/{filename}.png")
    plt.close()

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

# Hook to extract features from model.fc3
features_train = []
features_val = []
labels_train = []
labels_val = []

def hook(module, input, output):
    if model.training:
        features_train.append(input[0].detach().cpu().numpy())
        labels_train.append(input[0].detach().cpu().numpy())
    else:
        features_val.append(input[0].detach().cpu().numpy())
        labels_val.append(input[0].detach().cpu().numpy())

# Register the hook
model.fc3.register_forward_hook(hook)

# Training loop
for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = 0.0
    predicted_ages = []

    # Clear feature buffers
    features_train.clear()
    labels_train.clear()

    for idx, (features, sex, age) in enumerate(train_loader):
        features = features.unsqueeze(1).to(device)  # Add channel dimension
        sex = sex.to(device)
        age = age.to(device)
        optimizer.zero_grad()
        outputs = model(features, sex)
        train_outputs[train_indices[idx]] = outputs.item()

        loss = criterion(outputs.squeeze(), age)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    features_val.clear()
    labels_val.clear()

    with torch.no_grad():
        for idx, (features, sex, age) in enumerate(val_loader):
            features = features.unsqueeze(1).to(device)
            sex = sex.to(device)
            age = age.to(device)

            outputs = model(features, sex)
            loss = criterion(outputs.squeeze(), age)
            val_loss += loss.item()

            # Save the predicted age for the current validation sample
            val_outputs[val_indices[idx]] = outputs.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

    # ElasticNet Training and Validation
    X_train = np.concatenate(features_train)
    y_train = np.concatenate(labels_train)
    X_val = np.concatenate(features_val)
    y_val = np.concatenate(labels_val)

    elasticnet = ElasticNet(alpha=1.0, l1_ratio=0.5)  # Adjust hyperparameters as needed
    elasticnet.fit(X_train, y_train)

    # Evaluate ElasticNet on validation set
    y_pred_train = elasticnet.predict(X_train)
    y_pred_val = elasticnet.predict(X_val)

    train_mse = mean_absolute_error(y_train, y_pred_train)
    val_mse = mean_absolute_error(y_val, y_pred_val)

    print(f"ElasticNet - Train MSE: {train_mse:.4f}, Validation MSE: {val_mse:.4f}")

    # Save ElasticNet model and predictions
    with open(f"model_dumps/{sys.argv[1]}_elasticnet_model_epoch_{epoch+1}.pkl", "wb") as f:
        pickle.dump(elasticnet, f)

    pd.DataFrame({
        "True_Age": y_train,
        "Predicted_Age": y_pred_train
    }).to_csv(f"model_dumps/{sys.argv[1]}_elasticnet_train_predictions_epoch_{epoch+1}.csv", index=False)

    pd.DataFrame({
        "True_Age": y_val,
        "Predicted_Age": y_pred_val
    }).to_csv(f"model_dumps/{sys.argv[1]}_elasticnet_val_predictions_epoch_{epoch+1}.csv", index=False)

    # Save the last and best model logic remains unchanged
    # ...
    # Update epoch data and save the loss plot
    epoch_data.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })
    update_loss_plot(epoch_data, sys.argv[1])

    max_index = max(train_outputs.keys())
    # Create a DataFrame with NaN for all indices initially
    df_trn = pd.DataFrame(index=range(max_index + 1), columns=["Predicted_Age"])
    # Assign the values to their respective indices
    for index, value in train_outputs.items():
        df_trn.loc[index, "Predicted_Age"] = value
    # print (df_trn)

    df2 = df.copy()
    df2['Predicted_Age'] = df_trn['Predicted_Age']
    train_df = df2.loc[train_outputs.keys()]
    # print (train_df)
    train_df.to_csv(f"model_dumps/{sys.argv[1]}_predicted_ages_train.csv")

    max_index = max(val_outputs.keys())
    # Create a DataFrame with NaN for all indices initially
    df_pred = pd.DataFrame(index=range(max_index + 1), columns=["Predicted_Age"])
    # Assign the values to their respective indices
    for index, value in val_outputs.items():
        df_pred.loc[index, "Predicted_Age"] = value
    # print (df_pred)

    df1 = df.copy()
    df1['Predicted_Age'] = df_pred['Predicted_Age']
    test_df = df1.loc[val_outputs.keys()]
    # print (test_df)
    test_df.to_csv(f"model_dumps/{sys.argv[1]}_predicted_ages_val.csv")


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