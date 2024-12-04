import os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel

# Load the CSV file into a pandas DataFrame
csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
df = pd.read_csv(csv_path)

# Add a new column 'filepath' with the constructed file paths
df['filepath'] = df.apply(
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID']}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1
)

# Load pre-trained ViT model
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
model.eval()

# Define image transform for ViT input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# Directory to save processed images and features
os.makedirs("adni_storage/ADNI_features", exist_ok=True)

# Process each row in the DataFrame
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    filepath = row['filepath']
    image_title = f"{row['ImageID']}_{row['SubjectID']}"

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

            # Extract features for all sagittal slices
            features = []
            for slice_idx in range(data.shape[0]):
                slice_data = data[slice_idx, :, :]
                slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize

                # Transform slice for ViT input
                slice_tensor = transform(slice_data).unsqueeze(0)  # Add batch dimension

                # Extract features using ViT
                with torch.no_grad():
                    outputs = model(slice_tensor)
                    slice_features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Average over sequence
                    features.append(slice_features)

            # Save extracted features
            features = np.array(features)
            output_feature_path = f"adni_storage/ADNI_features/{image_title}_features.npy"
            np.save(output_feature_path, features)

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    else:
        print(f"File not found: {filepath}")
