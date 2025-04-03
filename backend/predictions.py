import os
import joblib
import pandas as pd
import numpy as np
import json
from datetime import timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the model
MODEL_PATH = "./backend/model_building/xgboost_model.joblib"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    # print("Model loaded successfully!")
else:
    raise FileNotFoundError("Model file not found!")

# Function to load dataset
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Function to preprocess SIM information
def classify_sim_info(sim_info):
    if isinstance(sim_info, str) and sim_info not in ['Unknown', ''] :
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
    columns_to_drop = ['Device number', 'Month', 'Office Date', 'Office Time In', 'Type', 'Final Status', 'Defect / Damage type', 'Responsible Party']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    df.rename(columns={'Product/Model #': 'Model'}, inplace=True)

    # Classify SIM information
    df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
    df.drop(columns=['sim_info'], inplace=True)

    if 'Number of Sim' in df.columns:
        if df['Number of Sim'] == 0:
            df["sim_info_status"] = 'uninserted'
        else:
            df["sim_info_status"] = 'inserted'

    # Convert date columns
    for col in ['last_boot_date', 'interval_date', 'active_date']:
        df[col] = df[col].astype(str).apply(convert_arabic_numbers)
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Set warranty status to 'Yes' or 'No' for devices within 30 days from the collection date (Feb 13, 2025)
    warranty_cutoff = pd.to_datetime('2025-02-13') - timedelta(days=30)

    # Assign 'Yes' or 'No' based on warranty conditions, only if the row is NaN
    df['Warranty'] = df.apply(
        lambda row: 'Yes' if pd.isna(row['Warranty']) and (
            row['last_boot_date'] >= warranty_cutoff or 
            row['interval_date'] >= warranty_cutoff or 
            row['active_date'] >= warranty_cutoff) else 'No',
        axis=1
    )
    
    # Compute time differences for churn calculation
    df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.days
    df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.days
    df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.days

    # Drop the original datetime columns after extracting features
    df.drop(columns=['last_boot_date', 'interval_date', 'active_date'], inplace=True)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df

# Function to predict churn on new data
def predict_churn(file, sheet):
    """Predict churn on new data from the specified Excel file and sheet."""
    # Load the dataset
    df = load_data(file, sheet)

    # Save a copy of the original data (before preprocessing)
    original_df = df.copy()

    # Preprocess the data
    df = preprocess_data(df)

    # Ensure the new data has the same features as the model
    X_unknown = df.drop(columns=['Churn'], errors='ignore')

    # Align the columns with the model's feature set (fill missing columns with 0)
    X_unknown = X_unknown.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict churn probabilities and predictions
    probabilities = model.predict_proba(X_unknown)[:, 1]  # Churn (class 1) probability
    predictions = model.predict(X_unknown)  # Churn predictions (0 or 1)

    # Add the predicted churn values to the DataFrame
    original_df['Churn_Predicted'] = predictions

    # Check if 'Device number' column exists
    if 'Device number' in original_df.columns:
        device_numbers = original_df['Device number'].copy()
    else:
        device_numbers = None  # Mark as missing

    # Generate the response with probabilities
    if device_numbers is not None:
        device_numbers = device_numbers.tolist()
        prediction_result = [{"Row Index": idx + 1, "Device number": device, 
                              "Churn Prediction": int(pred), "Churn Probability": float(prob)} 
                             for idx, (device, pred, prob) in enumerate(zip(device_numbers, predictions, probabilities))]
    else:
        prediction_result = [{"Row Index": idx + 1, "Churn Prediction": int(pred), "Churn Probability": float(prob)} 
                             for idx, (pred, prob) in enumerate(zip(predictions, probabilities))]

    # # Check if 'Churn' exists before printing
    # if 'Churn' in original_df.columns:
    #     print(original_df[['Churn', 'Churn_Predicted']].head())
    # else:
    #     print("Churn column is missing, skipping print.")

    return {"predictions": prediction_result}

def get_features():
    # Get feature importances from the model
    feature_importances = model.feature_importances_

    # Get feature names
    feature_names = model.feature_names_in_

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    # Convert 'Importance' to numeric and coerce errors (convert invalid values to NaN)
    importance_df['Importance'] = pd.to_numeric(importance_df['Importance'], errors='coerce')

    # Sort by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    return {"features": importance_df.to_dict(orient="records")}  # Convert to list of dicts

def evaluate_model(file, sheet):
    """Evaluate the model's performance on the provided dataset."""
    # Load the dataset
    df = load_data(file, sheet)

    # Save a copy of the original data (before preprocessing)
    original_df = df.copy()

    # Preprocess the data
    df = preprocess_data(df)

    # Ensure the new data has the same features as the model
    X_unknown = df.drop(columns=['Churn'], errors='ignore')

    # Align the columns with the model's feature set (fill missing columns with 0)
    X_unknown = X_unknown.reindex(columns=model.feature_names_in_, fill_value=0)

    # Get the true labels (Churn) from the original data (if available)
    y_true = original_df['Churn'] if 'Churn' in original_df.columns else None

    # If 'Churn' column is missing, we cannot calculate accuracy/precision/recall/F1
    if y_true is not None:
        # Drop NaN values from the true labels and corresponding rows in X_unknown
        y_true = y_true.dropna()
        X_unknown = X_unknown.loc[y_true.index]

        # Get predictions
        predictions = model.predict(X_unknown)

        # Calculate accuracy, precision, recall, and F1-score
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    else:
        return {"error": "True labels (Churn) are not available in the dataset."}
    
def get_confusion_matrix(file, sheet):
    """Compute and return the confusion matrix."""
    # Load the dataset
    df = load_data(file, sheet)

    # Save a copy of the original data (before preprocessing)
    original_df = df.copy()

    # Preprocess the data
    df = preprocess_data(df)

    # Ensure the new data has the same features as the model
    X_unknown = df.drop(columns=['Churn'], errors='ignore')

    # Align the columns with the model's feature set (fill missing columns with 0)
    X_unknown = X_unknown.reindex(columns=model.feature_names_in_, fill_value=0)

    # Get the true labels (Churn) from the original data (if available)
    y_true = original_df['Churn'] if 'Churn' in original_df.columns else None

    # If 'Churn' column is missing, we cannot calculate accuracy/precision/recall/F1
    if y_true is not None:
        # Drop NaN values from the true labels and corresponding rows in X_unknown
        y_true = y_true.dropna()
        X_unknown = X_unknown.loc[y_true.index]

        # Get predictions
        predictions = model.predict(X_unknown)

        conf_matrix = confusion_matrix(y_true, predictions)
        return {"confusion_matrix": conf_matrix.tolist()}  # Convert to list for easy JSON serialization
    else:
        return {"error": "Fail to return confusion matrix"}