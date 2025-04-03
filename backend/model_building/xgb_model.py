import pandas as pd
import numpy as np
import json
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Function to load dataset
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Function to preprocess SIM information
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

# Convert Arabic numerals to Western numerals
def convert_arabic_numbers(text):
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"
    return text.translate(str.maketrans(arabic_digits, western_digits)) if isinstance(text, str) else text

# Preprocessing function to clean and prepare the data
def preprocess_data(df):
    # Drop irrelevant columns
    columns_to_drop = ['Device number', 'Office Date', 'Office Time In', 'Type', 'Final Status', 'Defect / Damage type', 'Responsible Party']
    df.drop(columns=columns_to_drop, inplace=True)

    df.rename(columns={'Product/Model #': 'Model'}, inplace=True)

    # Classify SIM information
    df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
    df.drop(columns=['sim_info'], inplace=True)

    # Convert date columns
    for col in ['last_boot_date', 'interval_date', 'active_date']:
        df[col] = df[col].astype(str).apply(convert_arabic_numbers)
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Compute time differences for churn calculation
    df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.days
    df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.days
    df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.days

    # Drop the original datetime columns after extracting features
    df.drop(columns=['last_boot_date', 'interval_date', 'active_date'], inplace=True)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df

# Function to handle missing values
def handle_missing_values(df):
    # Separate rows with missing values for later prediction
    unknown_data = df[df.isnull().any(axis=1)]
    # Drop rows with missing values for model training
    df_cleaned = df.dropna()
    return df_cleaned, unknown_data

# Function to split data into features and target
def split_features_target(df_cleaned):
    X_cleaned = df_cleaned.drop(columns=['Churn'])
    y_cleaned = df_cleaned['Churn']
    return X_cleaned, y_cleaned

# Function to train the XGBoost model
def train_model(X_train, y_train):
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='f1')
    print(f"Mean Cross-Validation F1 Score: {cv_scores.mean():.4f}")

    # Train the model
    xgb_model.fit(X_train, y_train)
    return xgb_model

# Function to evaluate model performance
def evaluate_model(model, X_val, y_val, X_test, y_test):
    # Validate on the validation set
    y_val_pred = model.predict(X_val)
    print("Validation Set Classification Report:")
    print(classification_report(y_val, y_val_pred))

    # Test the model
    y_pred = model.predict(X_test)
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

# Function to save the trained model
def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Model saved successfully at {model_path}")

# Main function to run the entire workflow
def main():
    # Load the dataset
    df = load_data("UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")

    # Preprocess the data
    df_preprocessed = preprocess_data(df)

    # Define churn (0 if interval - activate < 30 days, else don't touch it)
    df_preprocessed['Churn'] = np.where(df_preprocessed['Churn'].isna() & (df_preprocessed['interval - activate'] < 30), 0, df_preprocessed['Churn'])

    # Handle missing values
    df_cleaned, unknown_data = handle_missing_values(df_preprocessed)

    # Split data into features and target
    X_cleaned, y_cleaned = split_features_target(df_cleaned)

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

    # Further split training set (80% train, 20% validation)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Handle class imbalance with SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_split, y_train_split)

    # Train the XGBoost model
    xgb_model = train_model(X_train_res, y_train_res)

    # Evaluate the model
    evaluate_model(xgb_model, X_val_split, y_val_split, X_test, y_test)

    # Predict missing data churn values
    X_unknown = unknown_data.drop(columns=['Churn'], errors='ignore')
    X_unknown = X_unknown[X_train_res.columns]  # Align columns with training set
    y_unknown = xgb_model.predict(X_unknown)

    # Assign predicted churn to unknown data
    unknown_data.loc[:, 'Churn_Predicted'] = y_unknown
    print("Rows with missing data and predicted churn values:")
    print(unknown_data[['Churn', 'Churn_Predicted']].head())

    # Save the trained model
    save_model(xgb_model, './backend/model_building/xgb_model.joblib')

if __name__ == "__main__":
    main()
