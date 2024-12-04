
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

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
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


predicted_ages = None
# Training loop
epochs = 30
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
