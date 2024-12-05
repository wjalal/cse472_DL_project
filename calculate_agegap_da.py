import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import make_interp_spline, interp1d
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Import SVM
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def calculate_lowess_yhat_and_agegap(dfres):
    dfres_agegap = dfres.copy()
    # calculate agegap using lowess of predicted vs chronological age from training cohort
    lowess = sm.nonparametric.lowess
    lowess_fit = lowess(dfres_agegap.Predicted_Age.to_numpy(), dfres_agegap.Age.to_numpy(), frac=0.8, it=3)
    lowess_fit_int = interp1d(lowess_fit[:,0], lowess_fit[:,1], bounds_error=False, kind='linear', fill_value=(0, 150)) 
    y_lowess = lowess_fit_int(dfres_agegap.Age)
    dfres_agegap["yhat_lowess"] = y_lowess
    # dfres_agegap["yhat_lowess"] = age_prediction_lowess(np.array(dfres_agegap.Age))
    if len(dfres_agegap.loc[dfres_agegap.yhat_lowess.isna()]) > 0:
        print("Could not predict lowess yhat in " + str(len(dfres_agegap.loc[dfres_agegap.yhat_lowess.isna()])) + " samples")
        dfres_agegap = dfres_agegap.dropna(subset="yhat_lowess")
    dfres_agegap["AgeGap"] = dfres_agegap["Predicted_Age"] - dfres_agegap["yhat_lowess"]
    dfres_agegap["AgeGap"] = dfres_agegap["AgeGap"].abs()
    return dfres_agegap

# Function to calculate MAE and R², and annotate the plot
def plot_with_metrics(data, x_col, y_col, hue_col, title, x_lim):
    # Calculate MAE and R²
    mae = mean_absolute_error(data[x_col], data[y_col])
    r2 = r2_score(data[x_col], data[y_col])
    
    # Create scatterplot
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, palette='coolwarm', hue_norm=(-12, 12))
    plt.xlim(*x_lim)
    plt.title(f"{title}\nMAE: {mae:.2f}, R²: {r2:.2f}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

# For training set
dfres_train = pd.read_csv("model_dumps/cnn_mx_elu_predicted_ages_train.csv", sep=",", index_col=0).reset_index()
dfres_train = calculate_lowess_yhat_and_agegap(dfres_train)

# Keep only the row with the smallest Age for each SubjectID
dfres_train = dfres_train.loc[dfres_train.groupby('SubjectID')['Age'].idxmin()]
dfres_train = dfres_train.reset_index(drop=True)

# For validation set
dfres_val = pd.read_csv("model_dumps/cnn_mx_elu_predicted_ages_val.csv", sep=",", index_col=0).reset_index()
dfres_val = calculate_lowess_yhat_and_agegap(dfres_val)

# Keep only the row with the smallest Age for each SubjectID
dfres_val = dfres_val.loc[dfres_val.groupby('SubjectID')['Age'].idxmin()]
dfres_val = dfres_val.reset_index(drop=True)


# Step 1: Encode categorical variables (for 'Sex' column)
dfres_train['Sex'] = dfres_train['Sex'].map({'M': 0, 'F': 1})
dfres_val['Sex'] = dfres_val['Sex'].map({'M': 0, 'F': 1})

# Step 2: Convert the 'Group' column to binary (AD vs not AD)
dfres_train['Group_binary'] = dfres_train['Group'].apply(lambda x: 1 if x == 'AD' else 0)
dfres_val['Group_binary'] = dfres_val['Group'].apply(lambda x: 1 if x == 'AD' else 0)

# Step 3: Initialize the LabelEncoder for the binary target 'Group_binary' column
y_train = dfres_train['Group_binary']
y_val = dfres_val['Group_binary']

print(f"Binary labels for training set: {y_train.unique()}")  # To verify the binary classification

# Step 4: Drop the original 'Group' column and prepare features for training
X_train = dfres_train[['AgeGap']]
X_val = dfres_val[['AgeGap']]

print(f"Features for training set:\n{X_train.head()}")

# Step 1: Prepare the data
X_train = torch.tensor(dfres_train[['AgeGap']].values, dtype=torch.float32)  # Feature
y_train = torch.tensor(dfres_train['Group_binary'].values, dtype=torch.float32)  # Target (binary)

X_val = torch.tensor(dfres_val[['AgeGap']].values, dtype=torch.float32)  # Feature
y_val = torch.tensor(dfres_val['Group_binary'].values, dtype=torch.float32)  # Target (binary)

# Step 2: Create Dataloader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 3: Define the Model with Attention Mechanism
class AttentionDNN(nn.Module):
    def __init__(self):
        super(AttentionDNN, self).__init__()
        self.dense1 = nn.Linear(1, 64)
        self.dropout1 = nn.Dropout(0.2)
        
        # Attention mechanism (simple scaled dot-product attention)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=1, batch_first=True)
        
        self.dense2 = nn.Linear(128, 32)  # 64 + 64 (Concatenated)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(32, 1)  # Binary output

    def forward(self, x):
        # First dense layer
        x = torch.relu(self.dense1(x))
        x = self.dropout1(x).unsqueeze(1)  # Add extra dimension for attention (batch_size, seq_len, feature)
        
        # Attention mechanism (requires 3 inputs: queries, keys, values)
        attn_output, _ = self.attention(x, x, x)
        
        # Concatenate attention output with original features
        x = torch.cat((x, attn_output), dim=-1)
        
        # Second dense layer
        x = torch.relu(self.dense2(x))
        x = self.dropout2(x)
        
        # Output layer (sigmoid for binary classification)
        x = torch.sigmoid(self.output(x.squeeze(1)))  # Remove extra dimension
        
        return x

# Step 4: Initialize the model, loss function, and optimizer
model = AttentionDNN()
criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))  # Reshaping labels for BCELoss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Step 6: Evaluate the model
model.eval()
y_pred_prob = []
y_true = []

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        y_pred_prob.append(outputs)
        y_true.append(labels)

y_pred_prob = torch.cat(y_pred_prob)
y_true = torch.cat(y_true)

# Convert to binary labels (0 or 1)
y_pred = (y_pred_prob > 0.5).float()

# Classification Report
class_names = ['Not AD', 'AD']
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
