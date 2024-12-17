import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
import os
import nibabel as nib 

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
# Load the CSV file into a pandas DataFrame
csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
df = pd.read_csv(csv_path).sample(n=120, random_state=420)
# df = df.sample(n=1000, random_state=69420)
print (df)
# Add a new column 'filepath' with the constructed file paths
df['filepath'] = df.apply(
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1
)


class My3DModel(tf.keras.Model):
    def __init__(self):
        super(My3DModel, self).__init__()
        
        self.conv1 = layers.Conv3D(64, 3, padding='same', activation='elu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv3D(64, 3, padding='same', activation='elu')
        self.bn2 = layers.BatchNormalization()
        
        # Attention layer
        self.attn = layers.MultiHeadAttention(num_heads=1, key_dim=64)

        self.pool = layers.MaxPooling3D(pool_size=(2, 2, 2))
        
        self.conv3 = layers.Conv3D(128, 3, padding='same', activation='elu')
        self.bn3 = layers.BatchNormalization()
        
        self.conv4 = layers.Conv3D(128, 3, padding='same', activation='elu')
        self.bn4 = layers.BatchNormalization()

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(128, activation='elu')
        self.dropout = layers.Dropout(0.5)
        self.output_layer = layers.Dense(1)

    def call(self, inputs):
        x, sex, age = inputs  # Unpack all three inputs
        
        # Reshape to add a depth dimension (1 in this case)
        x = tf.expand_dims(x, axis=-1)  # Shape: (batch_size, height, width, channels, depth)

        # Convolution and batch normalization
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Attention mechanism
        attn_output = self.attn(x, x)
        x = tf.concat([x, attn_output], axis=-1)
        
        # MaxPooling and Convolutions
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        
        # Flatten the feature maps
        x = self.flatten(x)
        
        # Concatenate `sex` and `age` to the flattened features before the fully connected layer
        x = tf.concat([x, sex, age], axis=-1)
        
        # Fully Connected Layer
        x = self.fc(x)
        x = self.dropout(x)
        
        # Output Layer
        return self.output_layer(x)

import numpy as np
import tensorflow as tf
import tensorflow as tf
import numpy as np

class ADNIDatasetLite:
    def __init__(self, image_list, sex_encoded, age_list):
        self.image_list = image_list
        self.sex_encoded = sex_encoded
        self.age_list = age_list

    def __len__(self):
        return len(self.age_list)

    def __getitem__(self, idx):
        image = np.array(self.image_list[idx], dtype=np.float32)  # Convert to numpy array
        sex = np.array(self.sex_encoded[idx], dtype=np.float32)
        age = np.array(self.age_list[idx], dtype=np.float32)
        return image, sex, age

# Prepare dataset and dataloaders
sex_encoded = df['Sex'].apply(lambda x: 0 if x == 'M' else 1).tolist()
age_list = df['Age'].tolist()

# Assuming image_list is populated with images

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

# Create Dataset object
dataset = ADNIDatasetLite(image_list, sex_encoded, age_list)
import numpy as np

# Ensure that all lists are the same length
print(f"Length of image_list: {len(image_list)}")
print(f"Length of sex_encoded: {len(sex_encoded)}")
print(f"Length of age_list: {len(age_list)}")

# Check if all lists are consistent
assert len(image_list) == len(sex_encoded) == len(age_list), "Data lists have mismatched lengths!"

# Assuming image_list is populated correctly
train_size = int(0.8 * len(image_list))  # Use len(image_list) if it matches the length of other lists
val_size = len(image_list) - train_size

# Now ensure safe indexing
train_data = [dataset[i] for i in range(min(train_size, len(dataset)))]
val_data = [dataset[i] for i in range(min(train_size, len(dataset)), len(dataset))]

# Continue with batching and model training...


# Function to batch data manually
def batch_data(data, batch_size):
    batched_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        images, sex, age = zip(*batch)  # Unzip data
        batched_data.append((
            np.stack(images),  # Stack images in a batch
            np.array(sex),     # Convert sex list to numpy array
            np.array(age)      # Convert age list to numpy array
        ))
    return batched_data

# Batch the data
train_batches = batch_data(train_data, batch_size=1)
val_batches = batch_data(val_data, batch_size=1)

# Define the model
model = My3DModel()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
              loss=tf.keras.losses.MeanSquaredError())

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Training loop
    for images, sex, age in train_batches:
        with tf.GradientTape() as tape:
            # Assuming the model takes (image, sex, age) as input
            predictions = model([images, sex, age], training=True)
            loss = model.loss(age, predictions)  # Loss on the age prediction
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Validation loop
    val_loss = 0
    for images, sex, age in val_batches:
        predictions = model([images, sex, age], training=False)
        val_loss += model.loss(age, predictions)

    val_loss /= len(val_batches)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

# After training, you can evaluate on the validation dataset
# Evaluation loop
val_loss = 0
for images, sex, age in val_batches:
    predictions = model([images, sex, age], training=False)
    val_loss += model.loss(age, predictions)

val_loss /= len(val_batches)
print(f"Final Validation Loss: {val_loss:.4f}")
