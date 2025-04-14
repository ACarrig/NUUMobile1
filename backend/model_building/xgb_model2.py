import re
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
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
                return 1 if carrier_name and carrier_name != 'Unknown' else 0
        except json.JSONDecodeError:
            return 0
    return 0

# Convert Arabic numerals to Western numerals
def convert_arabic_numbers(text):
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"
    return text.translate(str.maketrans(arabic_digits, western_digits)) if isinstance(text, str) else text

# Function to clean carrier info by removing everything in parentheses
def clean_carrier_label(label):
    # Use regular expression to remove content within parentheses and the parentheses themselves
    return re.sub(r'\s\([^)]+\)', '', label)

# Preprocessing function to clean and prepare the data
def preprocess_data(df):
    # Rename columns with small differences
    rename_dict = {
        'Product/Model #': 'Model',
        'last bootl date': 'last_boot_date',
        'interval date': 'interval_date',
        'activate date': 'active_date',
        'Sale Channel': 'Source',
    }
    
    df.rename(columns={key: value for key, value in rename_dict.items() if key in df.columns}, inplace=True)

    # Define columns to drop
    columns_to_drop = [
        'Device number', 'Month', 'Office Date', 'Office Time In', 'Final Status', 
        'Defect / Damage type', 'Responsible Party', 'Feedback', 'Slot 1', 'Slot 2', 
        'Verification', 'Spare Parts Used if returned', 'App Usage (s)', 
        'last boot - activate', 'last boot - interval'
    ]
    
    # Drop the columns that exist
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Check if 'Model' column exists before processing
    if "Model" in df.columns:
        # Normalize model names: remove spaces and use title case
        df["Model"] = df["Model"].str.strip().str.replace(" ", "", regex=True).str.lower()
        df["Model"] = df["Model"].replace({"budsa": "earbudsa", "budsb": "earbudsb"}).str.title()

    if 'Sim Country' in df.columns:
        # Apply the cleaning function to the feature's values
        df['Sim Country'] = df['Sim Country'].apply(clean_carrier_label)

    # Classify SIM information if column exists
    if 'sim_info' in df.columns:
        df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
        df.drop(columns=['sim_info'], inplace=True)
    elif 'Sim Card' in df.columns:
        df['sim_info_status'] = df['Sim Card'].apply(classify_sim_info)
        df.drop(columns=['Sim Card'], inplace=True)

    # Convert date columns if they exist
    date_columns = ['last_boot_date', 'interval_date', 'active_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(convert_arabic_numbers)
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Compute time differences for churn calculation
    if 'last_boot_date' in df.columns and 'active_date' in df.columns:
        df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.days
    if 'interval_date' in df.columns and 'last_boot_date' in df.columns:
        df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.days
    if 'interval_date' in df.columns and 'active_date' in df.columns:
        df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.days

    # Drop the original datetime columns after extracting features
    df.drop(columns=[col for col in date_columns if col in df.columns], inplace=True)

    # Create 'Churn' column based on 'Type'
    if 'Type' in df.columns:
        df['Churn'] = np.where(df['Type'] == 'Return', 1, np.where(df['Type'] == 'Repair', 0, np.nan))
        df.drop(columns=['Type'], inplace=True)

    df['Warranty'] = np.where(df['Churn'].isna() & (df['Warranty'] == "Yes"), 1, df['Warranty'])

    df.to_csv('./backend/model_building/data.csv', index=False)

    label_encoder = LabelEncoder()

    categorical_columns = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    for col in categorical_columns:
        if col in df.columns:
            # Handle mixed type columns
            df[col] = df[col].apply(str)  # Convert all values to strings first
            df[col] = label_encoder.fit_transform(df[col])

    return df

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
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='f1')
    print(f"Mean Cross-Validation F1 Score: {cv_scores.mean():.4f}")

    # Train the model
    xgb_model.fit(X_train, y_train)
    return xgb_model

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    # Test the model
    y_pred = model.predict(X_test)
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned (0)', 'Churned (1)'], 
                yticklabels=['Not Churned (0)', 'Churned (1)'])
    plt.title('Confusion Matrix for Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Main function to run the entire workflow
def main():
    # Load the dataset
    df = load_data("./backend/model_building/UW_Churn_Pred_Data.xls", sheet_name="Data")

    # Preprocess the data
    df_preprocessed = preprocess_data(df)

    # Define churn (0 if interval - activate < 30 days, else don't touch it)
    df_preprocessed['Churn'] = np.where(df_preprocessed['Churn'].isna() & (df_preprocessed['interval - activate'] < 30), 0, df_preprocessed['Churn'])

    # Handle missing values
    df_cleaned = df_preprocessed.dropna()

    # Split data into features and target
    X_cleaned, y_cleaned = split_features_target(df_cleaned)

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

    # Handle class imbalance with SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train the XGBoost model
    xgb_model = train_model(X_train_res, y_train_res)

    # Evaluate the model
    evaluate_model(xgb_model, X_test, y_test)

    # Save the trained model
    joblib.dump(xgb_model, './backend/model_building/xgb_model2.joblib')

if __name__ == "__main__":
    main()