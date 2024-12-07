import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import seaborn as sns

def set_random_seed(seed=69420):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(69420)
np.random.seed(69420)

temp_df_1 = pd.read_csv("model_dumps/cnn_mx_lrelu_predicted_ages_train.csv")
temp_df_2 = pd.read_csv("model_dumps/cnn_mx_lrelu_predicted_ages_val.csv")

# concatenate the two dataframes
temp_df = pd.concat([temp_df_1, temp_df_2])

# make a column for age gap, absolute difference between predicted and actual age

temp_df["AgeGap"] = abs(temp_df["Predicted_Age"] - temp_df["Age"])


# Load the data
csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
df = pd.read_csv(csv_path)
df['filepath'] = df.apply(
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID']}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1
)

df["AgeGap"] = df["ImageID"].map(temp_df.set_index("ImageID")["AgeGap"])

# convert the Group column to numeric
df['Group'] = df['Group'].map({'CN': 0, 'AD': 1, 'MCI': 2})
sex_encoded = df['Sex'].apply(lambda x: 0 if x == 'M' else 1).tolist()
age_gap_encoded = df['AgeGap'].tolist()

features_list = []
labels_list = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    filepath = row['filepath']
    image_title = f"{row['ImageID']}_{row['SubjectID']}"

    # Check if the feature file already exists
    feature_file_path = f"adni_storage/ADNI_features/{image_title}_features.npy"
    if os.path.exists(feature_file_path):
        # If file exists, load the features from the file
        features = np.load(feature_file_path)
        # scale the features
        features = StandardScaler().fit_transform(features)
        features_list.append(features)  # Flatten the features and add to the list
        labels_list.append(row['Group'])  # Add the corresponding age label

# Convert the lists to numpy arrays
features = np.array(features_list)
labels = np.array(labels_list)

#apply standard scaling to the features and age gap_encoded, sex_encoded
scaler = StandardScaler()
sex_encoded = scaler.fit_transform(np.array(sex_encoded).reshape(-1, 1)).flatten()
age_gap_encoded = scaler.fit_transform(np.array(age_gap_encoded).reshape(-1, 1)).flatten()



# Custom Dataset
class ADNIDataset(Dataset):
    def __init__(self, features_list, sex_list, age_gap_list, labels_list):
        self.features = features_list
        self.sex = sex_list
        self.labels = labels_list
        self.age_gap = age_gap_list
   

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.sex[idx], dtype=torch.float32),
            torch.tensor(self.age_gap[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
    
# Create Dataset and DataLoader
dataset = ADNIDataset(features_list, sex_encoded, age_gap_encoded, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


import importlib

module_name = sys.argv[1]  # Example: "my_model"
class_name = "AgePredictionCNNMulticlass"  # The class you want to import

try:
    # Dynamically import the module
    module = importlib.import_module(module_name)
    
    # Dynamically get the class
    AgePredictionCNNMulticlass = getattr(module, class_name)
    
    print(f"Successfully imported {class_name} from {module_name}.")

except ImportError:
    print(f"Module {module_name} could not be imported.")
except AttributeError:
    print(f"{class_name} does not exist in {module_name}.")

model = AgePredictionCNNMulticlass(features_list[0].shape, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


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
    loaded_accuracy = checkpoint["accuracy"]

    print(f"Loaded model from epoch {start_epoch} with best validation loss: {loaded_loss:.4f}")

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


def update_plot(epoch_data, filename):
    # Create a DataFrame from epoch data
    df = pd.DataFrame(epoch_data)
    df.to_csv(f"model_dumps/{filename}.csv", index=False)  # Save the data to CSV

    # Plot training and validation loss
    plt.figure(figsize=(10, 8))
    plt.plot(df['epoch'], df['train_loss'], label="Train Loss", marker="o")
    plt.plot(df['epoch'], df['val_loss'], label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"model_dumps/{filename}_loss.png")  # Save loss plot
    plt.close()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 8))
    plt.plot(df['epoch'], df['train_accuracy'], label="Train Accuracy", marker="o", color='green')
    plt.plot(df['epoch'], df['val_accuracy'], label="Validation Accuracy", marker="o", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"model_dumps/{filename}_accuracy.png")  # Save accuracy plot
    plt.close()


# Training parameters
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

start_epoch = 0
epochs = 100

best_accuracy = 0.0
val_accuracy = 0.0

# Train the model
for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for idx, (features, sex, age_gap, labels) in enumerate(train_loader):
        features = features.unsqueeze(1).to(device)  # Add channel dimension
        sex = sex.to(device)
        labels = labels.to(device)
        age_gap = age_gap.to(device)

        optimizer.zero_grad()
        outputs = model(features, sex, age_gap)

        loss = criterion(outputs, labels)  # CrossEntropyLoss expects raw logits
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)  # Get class predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = correct / total
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
    print(f"Epoch {epoch+1}/{epochs}, Train Accuracy: {train_accuracy:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    label_to_class = {0: 'CN', 1: 'AD', 2: 'MCI'}
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx, (features, sex, age_gap, labels) in enumerate(val_loader):
            features = features.unsqueeze(1).to(device)  # Add channel dimension
            sex = sex.to(device)
            labels = labels.to(device)
            age_gap = age_gap.to(device)

            outputs = model(features, sex, age_gap)
            loss = criterion(outputs, labels)  # CrossEntropyLoss expects raw logits
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Get class predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    val_loss /= len(val_loader)
    val_accuracy = correct / total

    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")
    print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}")

    all_preds_classes = [label_to_class[label] for label in all_preds]
    all_labels_classes = [label_to_class[label] for label in all_labels]
    
    

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "loss": val_loss,
        "accuracy": val_accuracy,
        "t_rng_st": torch.get_rng_state(),
        "n_rng_st": np.random.get_state(),
        "cuda_rng_st": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    with open(f"model_dumps/{sys.argv[1]}_last_model_with_metadata.pkl", "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Last model saved...")

    # Check if validation loss improved
    if  val_accuracy > best_accuracy:
        # best_loss = val_loss
        best_accuracy = val_accuracy
        print(f"Validation accuracy improved to {val_accuracy:.4f}. Saving the model...")
        with open(f"model_dumps/{sys.argv[1]}_best_model_with_metadata.pkl", "wb") as f:
            pickle.dump(checkpoint, f)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels_classes, all_preds_classes, labels=['CN', 'AD', 'MCI'])
        cm_display = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['CN', 'AD', 'MCI'], yticklabels=['CN', 'AD', 'MCI'])
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title(f"Confusion Matrix for Epoch {epoch+1}")
        plt.savefig(f"model_dumps/{sys.argv[1]}_conf_matrix.png")
        plt.close()
        print(f"Confusion matrix for epoch {epoch+1} saved.")

    epoch_data.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    update_plot(epoch_data, sys.argv[1])


    


