import pandas as pd
import numpy as np
import os


# Load the metadata CSV
df_ixi = pd.read_csv("ixi_storage/ixi_brainrotnet_metadata.csv")


df_ixi = pd.read_csv("ixi_storage/ixi_brainrotnet_metadata.csv")
df_ixi['uid'] = df_ixi['ImageID']

# Directory containing the files
ixi_dir = "ixi_storage/IXI-T1"

# List all files in the directory and get their absolute paths
file_list = [os.path.abspath(os.path.join(ixi_dir, f)) for f in os.listdir(ixi_dir) if os.path.isfile(os.path.join(ixi_dir, f))]

# Function to find the matching file path based on the first 6 characters
def find_matching_path(image_id, files):
    for file_path in files:
        file_name = os.path.basename(file_path)
        if file_name[:6] == image_id:
            return file_path
    return None



# Add the 'path' column by matching ImageID with file names
df_ixi['path'] = df_ixi['ImageID'].apply(lambda x: find_matching_path(x, file_list))

# Display a sample of the updated DataFrame
# Drop columns 'Sex' and 'ImageID'
df_ixi = df_ixi.drop(columns=['Sex', 'ImageID'])

# Rename the column 'Age' to 'age_at_scan'
df_ixi = df_ixi.rename(columns={'Age': 'age_at_scan'})

# print head of the updated DataFrame
# print(df_ixi.head())# Load the metadata CSV
df_ixi['Project'] = 'IXI'


df_fcon = pd.read_csv("fcon_storage/fcon1000_brainrotnet_metadata.csv")
df_fcon['uid'] = df_fcon['ImageID']
#drop first 8 characters from the 'ImageID' column, ad .nii.gz to the end and add     /home/nafiu/BasicData/cse472_DL_project/fcon_storage/fcon1000/ to the beginning
df_fcon['ImageID'] = '/home/nafiu/BasicData/cse472_DL_project/fcon_storage/fcon1000/' + df_fcon['ImageID'].str[8:] + '.nii.gz'
df_fcon = df_fcon.rename(columns={'ImageID': 'path'})
df_fcon = df_fcon.rename(columns={'Age': 'age_at_scan'})
df_fcon = df_fcon.drop(columns=['Sex'])
df_fcon['Project'] = 'FCON1000'

# print(df_fcon.head())# Load the metadata CSV

# new csv save
df_ixi.to_csv('ixi_storage/ixi_metadata.csv', index=False)


df_dlbs = pd.read_csv("dlbs_storage/dlbs_brainrotnet_metadata.csv")
df_dlbs['uid'] = df_dlbs['ImageID']
df_dlbs['ImageID'] = '/home/nafiu/BasicData/cse472_DL_project/dlbs_storage/DLBS/' + df_dlbs['ImageID'].str[4:] + '.nii.gz'

df_dlbs = df_dlbs.rename(columns={'ImageID': 'path'})
df_dlbs = df_dlbs.rename(columns={'Age': 'age_at_scan'})
df_dlbs = df_dlbs.drop(columns=['Sex'])
df_dlbs['Project'] = 'DLBS'


df_dlbs.to_csv('dlbs_storage/dlbs_metadata.csv', index=False)

# combine all the dataframes
df_combined = pd.concat([df_ixi, df_fcon, df_dlbs], ignore_index=True)
# save the combined dataframe to a csv file
df_combined.to_csv('combined_metadata.csv', index=False)
df_combined = pd.read_csv("combined_metadata.csv")  # Optionally use 'index_col=0' if the first column is the index
print(df_combined.head())

import uuid

# Assuming df_combined is already loaded
# Example: df_combined = pd.read_csv("combined_metadata.csv")

# Step 1: Add a unique 'indx' column (starting from 0)
df_combined['indx'] = range(len(df_combined))

# Step 2: Add a unique 'uid' column using UUIDs

# Step 3: Randomly assign 'train', 'dev', or 'test' based on the desired distribution
# Create a list of partitions: 60% train, 20% dev, 20% test
np.random.seed(42)  # For reproducibility
partition_labels = ['train'] * int(0.6 * len(df_combined)) + \
                   ['dev']   * int(0.2 * len(df_combined)) + \
                   ['test']  * int(0.2 * len(df_combined))

# If there are any remaining rows due to rounding errors, add them as 'train'
remaining = len(df_combined) - len(partition_labels)
partition_labels += ['train'] * remaining

# Shuffle the partition labels randomly
np.random.shuffle(partition_labels)

# Assign the shuffled partition labels to the DataFrame
df_combined['partition'] = partition_labels

# Save the updated DataFrame (optional)
df_combined.to_csv("combined_metadata.csv", index=False)

# Display the first few rows
print(df_combined.head())
print(df_combined.columns)

# Define the desired column order
desired_order = ['indx', 'Project', 'uid', 'age_at_scan', 'partition', 'path']

# Reorder the columns
df_combined = df_combined[desired_order]

# Save the updated DataFrame to a CSV (optional)
df_combined.to_csv("combined_metadata.csv", index=False)

# Verify the new column order
print(df_combined.head())
print(df_combined.columns)