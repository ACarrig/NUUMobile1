import re
import pandas as pd
import numpy as np
import json
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
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
    return df

# Function to classify SIM information as present (1) or not (0)
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
    return re.sub(r'\s\([^)]+\)', '', label)

# Preprocessing function to clean and prepare the data
def preprocess_data(df):
    # Rename columns to standardized names
    rename_dict = {
        'Product/Model #': 'Model',
        'last bootl date': 'last_boot_date',
        'interval date': 'interval_date',
        'activate date': 'active_date',
        'Sale Channel': 'Source',
    }
    
    # Only rename columns that exist in the dataframe
    df.rename(columns={key: value for key, value in rename_dict.items() if key in df.columns}, inplace=True)

    # Drop irrelevant or unnecessary columns if they exist
    columns_to_drop = [
        'Device number', 'imei1', 'Month', 'Office Date', 'Office Time In', 'Final Status', 
        'Defect / Damage type', 'Responsible Party', 'Feedback', 'Slot 1', 'Slot 2', 
        'Verification', 'Spare Parts Used if returned', 'App Usage (s)', 
        'last boot - activate', 'last boot - interval', 'activate'
    ]
    
    # Drop the columns that exist
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Check if 'Model' column exists before processing
    if "Model" in df.columns:
        # Normalize model names: remove spaces and use title case
        df["Model"] = df["Model"].str.strip().str.replace(" ", "", regex=True).str.lower().replace({"budsa": "earbudsa", "budsb": "earbudsb"}).str.title()

    # Clean the SIM country column values if present
    if 'Sim Country' in df.columns:
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

    # Compute time differences (in days) for churn calculation
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

    # If Churn column doesn't exist, add it with all values as NaN
    if 'Churn' not in df.columns:
        df['Churn'] = np.nan

    # Patch missing churn values using warranty information if available
    if 'Warranty' in df.columns:
        df['Warranty'] = np.where(df['Churn'].isna() & (df['Warranty'] == "Yes"), 1, df['Warranty'])

    # Encode all categorical string or boolean columns using LabelEncoder
    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].apply(str)
            df[col] = label_encoder.fit_transform(df[col])

    return df

# Function to get neural network model
def get_nnet_model(hidden_layer_sizes=(10,)*10, **kwargs):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        random_state=42,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        **kwargs
    )

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Model Classification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churned (0)', 'Churned (1)'], 
                yticklabels=['Not Churned (0)', 'Churned (1)'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def retrain_model(df):
    # Load and preprocess existing datasets
    df1 = load_data("./backend/model_building/UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
    df2 = load_data("./backend/model_building/UW_Churn_Pred_Data.xls", sheet_name="Data")
    
    df1_preprocessed = preprocess_data(df1)
    df2_preprocessed = preprocess_data(df2)
    df_new_preprocessed = preprocess_data(df)

    # Extract churn columns if they exist
    churn1 = df1_preprocessed['Churn'] if 'Churn' in df1_preprocessed.columns else pd.Series(dtype=int)
    churn2 = df2_preprocessed['Churn'] if 'Churn' in df2_preprocessed.columns else pd.Series(dtype=int)
    churn_new = df_new_preprocessed['Churn'] if 'Churn' in df_new_preprocessed.columns else pd.Series(dtype=int)

    # Get feature columns (exclude 'Churn') for each dataset
    features1 = df1_preprocessed.drop(columns=['Churn'], errors='ignore')
    features2 = df2_preprocessed.drop(columns=['Churn'], errors='ignore')
    features_new = df_new_preprocessed.drop(columns=['Churn'], errors='ignore')

    # Find common feature columns among all three feature sets
    common_features = list(set(features1.columns) & set(features2.columns) & set(features_new.columns))

    # Subset features to common columns
    features1_common = features1[common_features]
    features2_common = features2[common_features]
    features_new_common = features_new[common_features]

    # Concatenate features
    features_combined = pd.concat([features1_common, features2_common, features_new_common], ignore_index=True)

    # Concatenate churn series
    churn_combined = pd.concat([churn1, churn2, churn_new], ignore_index=True)

    # Add churn column back to features
    df_combined = features_combined.copy()
    df_combined['Churn'] = churn_combined

    # Fill missing Churn values based on your business logic
    df_combined['Churn'] = np.where(
        df_combined['Churn'].isna() & (df_combined['interval - activate'] < 30), 0, df_combined['Churn']
    )

    # Drop rows with missing values (you can also impute instead)
    df_cleaned = df_combined.dropna()

    # Separate target and features
    y = df_cleaned['Churn'].astype(int)
    X = df_cleaned.drop(columns=['Churn'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance with SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Create and train neural network model
    nnet_model = get_nnet_model(hidden_layer_sizes=(20, 20), alpha=0.01, learning_rate='adaptive')
    print("Training neural network...")
    nnet_model.fit(X_train_res, y_train_res)

    # Calibrate model
    print("\nCalibrating model...")
    calibrated_nnet = CalibratedClassifierCV(nnet_model, method='isotonic', cv='prefit')
    calibrated_nnet.fit(X_test, y_test)

    print("Model retrained successfully.")

    return calibrated_nnet, common_features

# Main function to run the neural network model
def main():
    # Load and preprocess data
    df1 = load_data("./backend/model_building/UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
    df2 = load_data("./backend/model_building/UW_Churn_Pred_Data.xls", sheet_name="Data")
    
    df1_preprocessed = preprocess_data(df1)
    df2_preprocessed = preprocess_data(df2)

    # Combine datasets
    common_cols = list(set(df1_preprocessed.columns) & set(df2_preprocessed.columns))
    df_combined = pd.concat([df1_preprocessed[common_cols], df2_preprocessed[common_cols]])
    
    # Fill in missing Churn values
    df_combined['Churn'] = np.where(
        df_combined['Churn'].isna() & (df_combined['interval - activate'] < 30), 0, df_combined['Churn']
    )

    # Remove any rows with missing values
    df_cleaned = df_combined.dropna()

    # Separate target and features
    y = df_cleaned['Churn']
    X = df_cleaned.drop(columns=['Churn'])
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Get neural network model
    nnet_model = get_nnet_model(hidden_layer_sizes=(20, 20), alpha=0.01, learning_rate='adaptive')

    # Train model
    print("Training neural network...")
    nnet_model.fit(X_train_res, y_train_res)

    # Calibrate the model
    print("\nCalibrating model...")
    calibrated_nnet = CalibratedClassifierCV(
        nnet_model,
        method='isotonic',
        cv='prefit'
    )
    calibrated_nnet.fit(X_test, y_test)

    # Final evaluation
    print("\nFinal Test Performance:")
    evaluate_model(calibrated_nnet, X_test, y_test)

    # Cross-validation scores
    cv_scores = cross_val_score(nnet_model, X, y, cv=5, scoring='balanced_accuracy')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {np.mean(cv_scores):.3f}")

    # Save model
    joblib.dump({
        'model': calibrated_nnet,
        'feature_names': list(X.columns)
    }, './backend/model_building/mlp_model.joblib')
    print("\nModel saved successfully.")

if __name__ == "__main__":
    main()