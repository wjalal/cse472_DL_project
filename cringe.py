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
from transformers import ViTForImageClassification
from transformers import DeiTForImageClassification
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
df_adni = df_adni.sort_values(by='Age', ascending=True).reset_index(drop=True).head(500)
# df_adni=df_adni.sample(n=400)

df = pd.concat ([
                 df_adni[['ImageID', 'Sex', 'Age', 'filepath']], 
                
                 ], ignore_index=True)

df['Age_Group'] = df['Age'].astype(int).apply(lambda x: f"{x:03d}"[:-1] + "0")
df['Age_Group'] = df['Age_Group'] + df['Sex']
print (df['Age_Group'].unique())
sex_encoded = df['Sex'].apply(lambda x: 0 if x == 'M' else 1).tolist()
age_list = df['Age'].tolist()
filepath_list = df['filepath'].tolist()
label_list = df['Age_Group'].tolist()

unique_labels = sorted(set(label_list))  # Ensure consistent ordering
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}  # Reverse mapping for decoding

# Convert labels to integers
numeric_labels = [label_to_idx[label] for label in label_list]
label_list = numeric_labels

roi = 160

# Transformation pipeline for ViT
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Convert to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize for ViT
])


# Function to extract 16 evenly spaced slices
def extract_slices(volume, num_slices=16):
    total_slices = volume.shape[0]
    indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    return volume[indices, :, :]  # Select slices

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


def crop_brain_volumes(brain_data):
    

        # Calculate bounding box from the brain volume
    min_indices, max_indices = calculate_bounding_box_from_volume(brain_data)

        # Crop the volume
    cropped_brain = brain_data[min_indices[0]:max_indices[0] + 1,
                                   min_indices[1]:max_indices[1] + 1,
                                   min_indices[2]:max_indices[2] + 1]
    return cropped_brain

# Function to preprocess data and dynamically expand slices while saving to disk
def preprocess_and_expand(dataset, transform, output_dir, num_slices=16):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    expanded_images, expanded_labels = [], []

    for filepath, label in tqdm(dataset, desc="Processing Slices"):
        # Check if all slice files already exist
        all_slices_exist = True
        slice_filenames = [
            os.path.join(output_dir, f"{os.path.basename(filepath)}_slice_{i}.pt")
            for i in range(num_slices)
        ]
        if not all(os.path.exists(slice_file) for slice_file in slice_filenames):
            all_slices_exist = False

        # Skip processing if all slices exist
        if all_slices_exist:
            expanded_images.extend(slice_filenames)  # Add existing file paths
            expanded_labels.extend([label] * num_slices)
            continue

        # Load NIfTI image only if slices are missing
        nii_img = nib.load(filepath)
        orig_ornt = io_orientation(nii_img.affine)
        ras_ornt = axcodes2ornt(("R", "A", "S"))
        ornt_trans = ornt_transform(orig_ornt, ras_ornt)
        data = nii_img.get_fdata()  # Load image data
        data = apply_orientation(data, ornt_trans)

        data = crop_brain_volumes(data)

        # Normalize and extract slices
        data = (data - data.min()) / (data.max() - data.min())
        slices = extract_slices(data, num_slices)

        # Transform each slice, save to file, and add to dataset
        for i, slice_data in enumerate(slices):
            slice_filename = slice_filenames[i]
            if not os.path.exists(slice_filename):
                transformed_slice = transform(slice_data)  # Transform slice
                torch.save(transformed_slice, slice_filename)  # Save to file
            expanded_images.append(slice_filename)  # Store file path
            expanded_labels.append(label)

    return expanded_images, expanded_labels
# Instantiate Dataset
vit_dataset = ADNIDatasetViT(filepath_list, label_list)

# Split Dataset
train_size = int(0.8 * len(vit_dataset))
val_size = len(vit_dataset) - train_size
generator = torch.Generator().manual_seed(universal_seed)
vit_train_dataset, vit_val_dataset = torch.utils.data.random_split(vit_dataset, [train_size, val_size], generator=generator)

# Create New Dataset with Filepaths
class ExpandedDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load the image from file
        image = torch.load(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

slice_count = 32

# Define output directory for slices
output_dir = f"processed_slices/{slice_count}"

# Preprocess and expand the training data
expanded_image_paths, expanded_labels = preprocess_and_expand(vit_train_dataset, transform, output_dir, num_slices=slice_count)

# Create Expanded Dataset and DataLoader
expanded_train_dataset = ExpandedDataset(expanded_image_paths, expanded_labels)
expanded_train_loader = DataLoader(expanded_train_dataset, batch_size=8, shuffle=True)


# Load ViT model
num_classes = df['Age_Group'].nunique()  # Number of unique Age_Groups
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True, 
)

model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Function to save checkpoint
def save_checkpoint(epoch, model, optimizer, path=f"model_dumps/vit_train_checkpoint_{slice_count}.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at epoch {epoch+1}")

vit_train_epochs = 6
model.train()
start_epoch = 0

for epoch in range(start_epoch, vit_train_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(expanded_train_loader, desc=f"Epoch {epoch+1}/{vit_train_epochs}"):
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(pixel_values=inputs)  # ViT expects `pixel_values`
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.logits, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(expanded_train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{vit_train_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    # sav the accuracy and loss for each epoch in a csv file
    with open(f'vit_train_metrics_{slice_count}.csv', 'a') as f:
        f.write(f"{epoch+1},{epoch_loss},{epoch_accuracy}\n")
    
    save_checkpoint(epoch, model, optimizer, path=f"model_dumps/vit_train_checkpoint_{slice_count}.pth")
    # model.to(device)  # Move back to GPU
    gc.collect()

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Convert to RGB directly
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

torch.cuda.empty_cache()  # Free GPU memory

# To store features and labels
features_list = []
labels_list = []

os.makedirs(f"adni_storage/ADNI_features/train_e{vit_train_epochs}/{slice_count}/", exist_ok=True)
# Process each row in the DataFrame
for _, row in tqdm(df_adni.iterrows(), total=len(df_adni), desc="Processing images"):
    filepath = row['filepath']
    image_title = f"{row['ImageID'][4:]}_{row['SubjectID']}"

    # Check if the feature file already exists
    feature_file_path = f"adni_storage/ADNI_features/train_e{vit_train_epochs}/{slice_count}/{image_title}_features.npy"
    if os.path.exists(feature_file_path):
        # If file exists, load the features from the file
        features = np.load(feature_file_path)
        features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
        
        from PIL import Image
        # Normalize the array to 0-255 for grayscale image
        data_normalized = ((features - np.min(features)) / (np.max(features) - np.min(features)) * 255).astype(np.uint8)
        data_normalized = np.repeat(data_normalized, 4, axis=0)
        # Create an image from the array
        img = Image.fromarray(np.transpose(data_normalized), mode='L')  # 'L' mode for grayscale
        # Save the image
        # img.save(f"adni_storage/ADNI_features/train_e{vit_train_epochs}/{slice_count}/featuremaps/{image_title}_fm.png")

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

                data = crop_brain_volumes(data)

                # Resample the volume to 160 slices (if required)
                data = resample_nifti(data, target_slices=160)

                # Extract features for all sagittal slices
                features = []
                for slice_idx in range(data.shape[0]):
                    slice_data = data[slice_idx, :, :]
                    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize

                    # Transform slice for ViT input
                    slice_tensor = transform(slice_data).unsqueeze(0).to(device)  # Add batch dimension and move to GPU

                    # Extract features using ViT
                    with torch.no_grad():
                        # #outputs = model(slice_tensor)
                        # slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Move output back to CPU
                        slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
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


batch_size = 1

print (features_list[0].shape)

dataset = ADNIDataset(features_list, sex_encoded, age_list)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
generator.manual_seed(universal_seed)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
val_indices = val_dataset.indices
train_indices = train_dataset.indices

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_outputs = {}
train_outputs = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

print (features_list[0].shape)
model = AgePredictionCNN((1, features_list[0].shape[0], features_list[0].shape[1])).to(device)
criterion = nn.L1Loss()
eval_crit = nn.L1Loss()
# adamw 
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
best_loss = np.inf  # Initialize the best loss to infinity
start_epoch = 0
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


predicted_ages = None
# Training loop
epochs = int(sys.argv[3])

# Initialize lists to track loss
filename = sys.argv[1] 
csv_file = f"model_dumps/mix/{slice_count}/{filename}.csv"

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
        # Store the output for each sample in the batch
        for i in range(outputs.size(0)):
            train_outputs[train_indices[idx * batch_size + i]] = outputs[i].item()

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
        for idx, (features, sex, age) in enumerate(val_loader):
            features = features.unsqueeze(1).to(device)
            sex = sex.to(device)
            age = age.to(device)

            outputs = model(features, sex)
            loss = eval_crit(outputs.squeeze(), age)
            val_loss += loss.item()

            # Save the predicted age for the current validation sample
            for i in range(outputs.size(0)):
                val_outputs[val_indices[idx * batch_size + i]] = outputs[i].item()
            # val_outputs[val_indices[idx]] = outputs.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")


    # Check if validation loss improved
    if val_loss < best_loss:
        best_loss = val_loss
        print(f"Validation loss improved to {best_loss:.4f}. Saving model...")
       
