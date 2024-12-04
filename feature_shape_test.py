import os
import numpy as np

# Path to the directory containing the .npy files
features_dir = "adni_storage/ADNI_features/"

# List to store file paths and their shapes
shapes = []

# Iterate over all .npy files in the directory
for filename in os.listdir(features_dir):
    if filename.endswith(".npy"):
        # Load the .npy file
        filepath = os.path.join(features_dir, filename)
        data = np.load(filepath)
        
        # Store the shape and filename
        shapes.append((filename, data.shape))

# Sort the list by the first dimension (shape[0])
shapes.sort(key=lambda x: x[1][0])

# Print the sorted shapes
for filename, shape in shapes:
    print(f"{filename}: {shape}")

print(shapes[0])