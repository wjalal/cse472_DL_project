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

# Step 5: Train the Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X_train)

# Step 6: Predict probabilities and classify
# GMM provides probabilities for each component (class). Choose the class with the highest probability.
y_pred_prob = gmm.predict_proba(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)  # Get the class with the highest probability

# Get the class labels (binary: 0 = not AD, 1 = AD)
class_names = ['Not AD', 'AD']

# Step 7: Evaluate the GMM model
print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_val, y_pred)

# Create a heatmap with seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Step 8: (Optional) Evaluate overall accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.2f}")