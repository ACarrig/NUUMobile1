import pandas as pd
import numpy as np
import json
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset from the specified Excel sheet
df = pd.read_excel("UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

# List of columns to drop (irrelevant or non-numeric)
columns_to_drop = ['Device number', 'Product/Model #', 'Office Date', 'Office Time In', 'Final Status','Defect / Damage type', 'Responsible Party']
df = df.drop(columns=columns_to_drop)
print(f"Columns after dropping: {list(df.columns)}")

# Function to classify SIM information as 'inserted' or 'uninserted'
def classify_sim_info(sim_info):
    if isinstance(sim_info, str) and sim_info != 'Unknown' and sim_info != '':
        try:
            parsed = json.loads(sim_info)  # Parse JSON-formatted string
            if isinstance(parsed, list) and parsed:
                carrier_name = parsed[0].get('carrier_name', None)
                if carrier_name and carrier_name != 'Unknown':
                    return 'inserted'
            return 'uninserted'
        except json.JSONDecodeError:
            return 'uninserted'  # Handle malformed JSON
    else:
        return 'uninserted'  # Default to 'uninserted' for missing or unknown values

# Apply classification function to the 'sim_info' column
df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)

# Drop the original 'sim_info' column as it is no longer needed
df = df.drop(columns=['sim_info'])
print(f"Data after sim_info classification and drop: {df.head()}")

# Function to convert Arabic numerals to Western numerals
def convert_arabic_numbers(text):
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"
    translation_table = str.maketrans(arabic_digits, western_digits)
    return text.translate(translation_table) if isinstance(text, str) else text

# Convert Arabic numerals and parse date columns
for col in ['last_boot_date', 'interval_date', 'active_date']:
    df[col] = df[col].astype(str).apply(convert_arabic_numbers)  # Convert Arabic numerals
    df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert to datetime format
print(f"Converted date columns: {df[['last_boot_date', 'interval_date', 'active_date']].head()}")

# Compute time differences in days
df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.total_seconds() / (60 * 60 * 24)
df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.total_seconds() / (60 * 60 * 24)
df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.total_seconds() / (60 * 60 * 24)
print(f"First few rows after calculating date differences: {df[['last_boot - activate', 'interval - last_boot', 'interval - activate']].head()}")

# Define the churn column: if interval - activate > 30 days, mark as churned (1), else not churned (0)
df['Churn'] = (df['interval - activate'] > 30).astype(int)

# Drop datetime columns as they have been processed into numerical values
df = df.drop(columns=['interval_date', 'last_boot_date', 'active_date'])
print(f"Churn column created: {df[['Churn']].head()}")

# Convert categorical columns to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)
print(f"Data after one-hot encoding: {df.head()}")

# Separate rows with missing values
unknown_data = df[df.isnull().any(axis=1)]  # Rows with NaN values
df_cleaned = df.dropna()  # Rows without NaN values
print(f"Rows with missing data: {unknown_data.shape[0]}, Rows without missing data: {df_cleaned.shape[0]}")

# Separate features and target variable
X_cleaned = df_cleaned.drop(columns=['Churn'])  # Features
y_cleaned = df_cleaned['Churn']  # Target

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Further split training set into training (80%) and validation (20%) sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(f"Training split size: {X_train_split.shape[0]}, Validation split size: {X_val_split.shape[0]}")

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_split, y_train_split)

# Initialize and train the Random Forest model
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf.fit(X_train_res, y_train_res)
print("Random Forest model trained.")

# Validate the model on the validation set
y_val_pred = rf.predict(X_val_split)
print("Validation Classification Report:")
print(classification_report(y_val_split, y_val_pred))

# Test the model on the test set
y_pred = rf.predict(X_test)

# Print the classification report for test set
print("Test Set Classification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix for the test set
cm = confusion_matrix(y_test, y_pred)

# Print and plot confusion matrix
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned (0)', 'Churned (1)'], yticklabels=['Not Churned (0)', 'Churned (1)'])
plt.title('Confusion Matrix for Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Predict churn values for the missing data rows
X_unknown = unknown_data.drop(columns=['Churn'])  # Features for prediction
y_unknown = rf.predict(X_unknown)  # Predicted churn values

# Assign predicted churn values back to the unknown data
unknown_data.loc[:, 'Churn_Predicted'] = y_unknown  # Avoid SettingWithCopyWarning

# Display predicted churn values for rows with missing data
print("Rows with missing data and predicted churn values:")
print(unknown_data[['Churn', 'Churn_Predicted']].head())

# Extract feature importances from the trained model
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the top 10 most important features
print("Top 10 most important features:")
for i in range(10):
    print(f"{X_cleaned.columns[indices[i]]}: {importances[indices[i]]}")

# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(10), importances[indices[:10]], align="center")
plt.yticks(range(10), [X_cleaned.columns[i] for i in indices[:10]])
plt.xlabel('Relative Importance')
plt.show()

# Save the trained Random Forest model to a file for later use
joblib.dump(rf, 'random_forest_model.joblib')
print("Model saved successfully.")
