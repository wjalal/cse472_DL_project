import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import make_interp_spline, interp1d
import matplotlib.pyplot as plt

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

# For training set
dfres_train = pd.read_csv("predicted_ages_train.csv", sep=",", index_col=0).reset_index()
dfres_train = calculate_lowess_yhat_and_agegap(dfres_train)

# Keep only the row with the smallest Age for each SubjectID
dfres_train = dfres_train.loc[dfres_train.groupby('SubjectID')['Age'].idxmin()]

# Rebuild the index
dfres_train = dfres_train.reset_index(drop=True)

# Plot the scatterplot for the train set
toplot = dfres_train.sort_values("Age")
sns.scatterplot(data=toplot, x="Age", y="Predicted_Age", 
                hue="AgeGap", palette='coolwarm', hue_norm=(-12, 12))
plt.xlim(40, 100)
plt.title("Age gap predictions (Train Set)")
plt.show()

print(dfres_train)

# For validation set
dfres_val = pd.read_csv("predicted_ages_val.csv", sep=",", index_col=0).reset_index()
dfres_val = calculate_lowess_yhat_and_agegap(dfres_val)

# Keep only the row with the smallest Age for each SubjectID
dfres_val = dfres_val.loc[dfres_val.groupby('SubjectID')['Age'].idxmin()]

# Rebuild the index
dfres_val = dfres_val.reset_index(drop=True)

# Plot the scatterplot for the validation set
toplot = dfres_val.sort_values("Age")
sns.scatterplot(data=toplot, x="Age", y="Predicted_Age", 
                hue="AgeGap", palette='coolwarm', hue_norm=(-12, 12))
plt.xlim(50, 100)
plt.title("Age gap predictions (Validation Set)")
plt.show()

print(dfres_val)

# Swapped Box plot: AgeGap vs Group for the training set
plt.figure(figsize=(10, 6))
sns.boxplot(data=dfres_train, x="AgeGap", y="Group", palette='coolwarm')
plt.title("AgeGap vs Group - Training Set")
plt.xlabel("AgeGap")
plt.ylabel("Group")
plt.show()

# Swapped Box plot: AgeGap vs Group for the validation set
plt.figure(figsize=(10, 6))
sns.boxplot(data=dfres_val, x="AgeGap", y="Group", palette='coolwarm')
plt.title("AgeGap vs Group - Validation Set")
plt.xlabel("AgeGap")
plt.ylabel("Group")
plt.show()




# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC  # Import SVM
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Step 1: Encode categorical variables (for 'Sex' column)
# dfres_train['Sex'] = dfres_train['Sex'].map({'M': 0, 'F': 1})
# dfres_val['Sex'] = dfres_val['Sex'].map({'M': 0, 'F': 1})

# # Step 2: Initialize the LabelEncoder for the target 'Group' column
# label_encoder = LabelEncoder()

# # Step 3: Fit and transform the 'Group' column for training and validation datasets
# y_train = label_encoder.fit_transform(dfres_train['Group'])
# y_val = label_encoder.transform(dfres_val['Group'])

# print(y_train)

# # Step 4: Drop the original 'Group' column and prepare features for training
# X_train = dfres_train[['AgeGap']]
# X_val = dfres_val[['AgeGap']]

# print(X_train)


# # Step 5: Train the Support Vector Machine (SVM) model
# svm_model = SVC(kernel='linear')  # You can change the kernel if needed
# svm_model.fit(X_train, y_train)

# # Step 6: Evaluate the model
# y_pred = svm_model.predict(X_val)

# # Get class names (mapping the integer-encoded labels back to original categories)
# class_names = label_encoder.classes_

# print("Classification Report:")
# print(classification_report(y_val, y_pred, target_names=class_names))

# print("Confusion Matrix:")
# print(confusion_matrix(y_val, y_pred))

# # Compute confusion matrix
# conf_matrix = confusion_matrix(y_val, y_pred)

# # Create a heatmap with seaborn
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Import SVM
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# Step 5: Train the Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear')  # You can change the kernel if needed
svm_model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = svm_model.predict(X_val)

# Get the class labels (binary: 0 = not AD, 1 = AD)
class_names = ['Not AD', 'AD']

print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)

# Create a heatmap with seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
