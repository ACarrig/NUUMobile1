import pandas as pd
import numpy as np
import json
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df = pd.read_excel("UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

print("Columns in the dataset:", df.info())

# Drop irrelevant columns
columns_to_drop = ['Device number', 'Product/Model #', 'Office Date', 'Office Time In', 'Type', 'Final Status', 'Defect / Damage type', 'Responsible Party']
df.drop(columns=columns_to_drop, inplace=True)

# Function to classify SIM information
def classify_sim_info(sim_info):
    if isinstance(sim_info, str) and sim_info not in ['Unknown', '']:
        try:
            parsed = json.loads(sim_info)
            if isinstance(parsed, list) and parsed:
                carrier_name = parsed[0].get('carrier_name', None)
                return 'inserted' if carrier_name and carrier_name != 'Unknown' else 'uninserted'
        except json.JSONDecodeError:
            return 'uninserted'
    return 'uninserted'

df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
df.drop(columns=['sim_info'], inplace=True)

# Function to convert Arabic numerals to Western numerals
def convert_arabic_numbers(text):
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"
    return text.translate(str.maketrans(arabic_digits, western_digits)) if isinstance(text, str) else text

# Convert date columns
for col in ['last_boot_date', 'interval_date', 'active_date']:
    df[col] = df[col].astype(str).apply(convert_arabic_numbers)
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Compute time differences
df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.days
df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.days
df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.days

# Define churn (1 if interval - activate > 30 days, else 0)
df['Churn'] = (df['interval - activate'] > 30).astype(int)

# Drop date columns
df.drop(columns=['interval_date', 'last_boot_date', 'active_date'], inplace=True)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)

# Handle missing values
unknown_data = df[df.isnull().any(axis=1)]
df_cleaned = df.dropna()

# Separate features and target
X_cleaned = df_cleaned.drop(columns=['Churn'])
y_cleaned = df_cleaned['Churn']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Further split training set (80% train, 20% validation)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_split, y_train_split)

# Drop highly correlated features
corr_matrix = X_train_res.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_train_res.drop(columns=to_drop, inplace=True)
X_val_split.drop(columns=to_drop, inplace=True)
X_test.drop(columns=to_drop, inplace=True)

# Train Random Forest Classifier with regularization
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    bootstrap=True,
    max_samples=0.7,
    random_state=42,
    class_weight='balanced_subsample'
)

# Perform cross-validation
cv_scores = cross_val_score(rf, X_train_res, y_train_res, cv=5, scoring='f1')
print(f"Mean Cross-Validation F1 Score: {cv_scores.mean():.4f}")

# Train the model
rf.fit(X_train_res, y_train_res)
print("Random Forest model trained.")

# Validate on the validation set
y_val_pred = rf.predict(X_val_split)
print("Validation Set Classification Report:")
print(classification_report(y_val_split, y_val_pred))

# Test the model
y_pred = rf.predict(X_test)

# Test set evaluation
print("Test Set Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned (0)', 'Churned (1)'], 
            yticklabels=['Not Churned (0)', 'Churned (1)'])
plt.title('Confusion Matrix for Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Predict missing data churn values
X_unknown = unknown_data.drop(columns=['Churn'] + to_drop, errors='ignore')
y_unknown = rf.predict(X_unknown)
unknown_data.loc[:, 'Churn_Predicted'] = y_unknown

# Display predicted churn values
print("Rows with missing data and predicted churn values:")
print(unknown_data[['Churn', 'Churn_Predicted']].head())

# Extract and plot feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 5 most important features:")
for i in range(5):
    print(f"{X_cleaned.columns[indices[i]]}: {importances[indices[i]]}")

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(5), importances[indices[:5]], align="center")
plt.yticks(range(5), [X_cleaned.columns[i] for i in indices[:5]])
plt.xlabel('Relative Importance')
plt.show()

# Save the trained model
joblib.dump(rf, './backend/model_building/random_forest_model.joblib')
print("Model saved successfully.")
