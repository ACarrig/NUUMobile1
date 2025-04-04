import os
import joblib
import pandas as pd
import numpy as np
import json
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')

# Load the model
# MODEL_PATH = "./backend/model_building/rf_model.joblib"
MODEL_PATH = "./backend/model_building/xgb_model.joblib"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    # print("Model loaded successfully!")
else:
    raise FileNotFoundError("Model file not found!")

# Function to load dataset
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
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
    # Columns to drop
    columns_to_drop = ['Device number', 'Office Date', 'Office Time In', 
                       'Type', 'Final Status', 'Defect / Damage type', 'Responsible Party']

    # Check if columns exist in the DataFrame before dropping them
    columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
    
    # Drop the existing columns
    df.drop(columns=columns_to_drop_existing, inplace=True)

    if 'Product/Model #' in df.columns:
        df.rename(columns={'Product/Model #': 'Model'}, inplace=True)

    # Classify SIM information if the column exists
    if 'sim_info' in df.columns:
        df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
        df.drop(columns=['sim_info'], inplace=True)
    # else:
    #     print("'sim_info' column is missing, skipping classification.")

    if 'Number of Sim' in df.columns:
        if (df['Number of Sim'] == 0).any():
            df["sim_info_status"] = 'uninserted'
        else:
            df["sim_info_status"] = 'inserted'

    # Convert date columns
    for col in ['last_boot_date', 'interval_date', 'active_date']:
        if col in df.columns:  # Check if the date column exists before processing
            df[col] = df[col].astype(str).apply(convert_arabic_numbers)
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Compute time differences for churn calculation
    df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.days if 'last_boot_date' in df.columns and 'active_date' in df.columns else np.nan
    df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.days if 'interval_date' in df.columns and 'last_boot_date' in df.columns else np.nan
    df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.days if 'interval_date' in df.columns and 'active_date' in df.columns else np.nan

    if 'active_date' in df.columns:
        df['Warranty'] = np.where(df['Warranty'].isna() & ((datetime(2025, 2, 13) - df['active_date']).dt.days < 30), 'Yes', df['Warranty'])

    # Drop date columns after creating churn
    df.drop(columns=['interval_date', 'last_boot_date', 'active_date'], inplace=True, errors='ignore')

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

    # Drop columns that are not part of the model's feature set
    X_unknown = X_unknown.loc[:, X_unknown.columns.isin(model.feature_names_in_)]

    # Add missing columns (set default value to 0 or appropriate default)
    missing_columns = set(model.feature_names_in_) - set(X_unknown.columns)
    for col in missing_columns:
        X_unknown[col] = 0  # You can adjust the default value if needed

    # Reorder the columns to match the model's expected feature set
    X_unknown = X_unknown[model.feature_names_in_]

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

    # Drop columns that are not part of the model's feature set
    X_unknown = X_unknown.loc[:, X_unknown.columns.isin(model.feature_names_in_)]

    # Add missing columns (set default value to 0 or appropriate default)
    missing_columns = set(model.feature_names_in_) - set(X_unknown.columns)
    for col in missing_columns:
        X_unknown[col] = 0  # Default value for missing columns

    # Reorder the columns to match the model's expected feature set
    X_unknown = X_unknown[model.feature_names_in_]

    # Predict churn probabilities and predictions
    probabilities = model.predict_proba(X_unknown)[:, 1]  # Churn (class 1) probability
    predictions = model.predict(X_unknown)  # Churn predictions (0 or 1)

    # Add the predicted churn values to the DataFrame
    original_df['Churn_Predicted'] = predictions

    # Check if 'Churn' column exists for evaluation
    if 'Churn' not in original_df.columns:
        return {"error": "True labels (Churn) are not available in the dataset."}

    # Split features and target for evaluation
    y_true = original_df['Churn']

    # Drop rows with missing true labels for Churn
    original_df = original_df.dropna(subset=['Churn'])

    # Align features with true labels after dropping NaNs
    y_true = original_df['Churn']
    predictions = original_df['Churn_Predicted']

    # Ensure that the predictions and true labels have the same length
    if len(y_true) != len(predictions):
        return {"error": f"Mismatch in the number of rows: {len(y_true)} true labels vs {len(predictions)} predictions."}

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)

    # Confusion Matrix
    cm = confusion_matrix(y_true, predictions)

    # Generate a plot of the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save the plot to a BytesIO object to convert it to a base64 string
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    # Close the plot
    plt.close()

    # Return results as JSON with the base64-encoded confusion matrix image
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix_image": img_base64  # Base64-encoded image of confusion matrix
    }
