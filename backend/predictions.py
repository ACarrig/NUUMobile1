import os
import joblib
import pandas as pd
import numpy as np
import json

# Load the model once at startup
MODEL_PATH = "./backend/model_building/random_forest_model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
else:
    raise FileNotFoundError("Model file not found!")

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

# Convert Arabic numerals to Western numerals
def convert_arabic_numbers(text):
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"
    return text.translate(str.maketrans(arabic_digits, western_digits)) if isinstance(text, str) else text

# Function to preprocess the data
def preprocess_data(df, model_columns=None):
    # Ensure column names are consistent
    df.columns = df.columns.str.strip()

    # Classify SIM information
    if 'sim_info' in df.columns:
        df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
        df.drop(columns=['sim_info'], inplace=True)

    # Convert Arabic numerals
    date_columns = ['last_boot_date', 'interval_date', 'active_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(convert_arabic_numbers)
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Compute time differences safely
    if all(col in df.columns for col in ['last_boot_date', 'active_date']):
        df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.days

    if all(col in df.columns for col in ['interval_date', 'last_boot_date']):
        df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.days

    if all(col in df.columns for col in ['interval_date', 'active_date']):
        df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.days

    # Drop original date columns
    df.drop(columns=[col for col in date_columns if col in df.columns], inplace=True)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Align columns with the model (if provided)
    if model_columns is not None:
        missing_cols = set(model_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0  # Add missing columns with neutral value

        extra_cols = set(df.columns) - set(model_columns)
        if extra_cols:
            print(f"Warning: Extra columns found {extra_cols}. Removing them.")
            df = df.drop(columns=extra_cols)

        # Ensure correct column order
        df = df[model_columns]

    return df

# Function to predict churn on new data
def predict_churn(file, sheet):
    """Predict churn on new data from the specified Excel file and sheet."""
    # Load the file & sheet
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, sheet_name=sheet)

    # Ensure the columns are aligned with the model's expected input
    model_columns = model.feature_names_in_
    df_processed = preprocess_data(df, model_columns=model_columns)

    # Predict churn probabilities
    probabilities = model.predict_proba(df_processed)[:, 1]  # Churn (class 1) probability
    predictions = model.predict(df_processed)  # Churn predictions (0 or 1)

    # Convert to Python native types (int, float) to avoid JSON serialization issues
    probabilities = probabilities.tolist()  # Convert numpy array to list
    predictions = predictions.tolist()  # Convert numpy array to list

    # Check if 'Device number' column exists
    if 'Device number' in df.columns:
        device_numbers = df['Device number'].copy()
    else:
        device_numbers = None  # Mark as missing

    # Generate the response with probabilities
    if device_numbers is not None:
        device_numbers = device_numbers.tolist()
        prediction_result = [{"Row Index": idx + 1, "Device number": device, "Churn Prediction": 1-pred, "Churn Probability": 1-prob} 
                             for idx, (device, pred, prob) in enumerate(zip(device_numbers, predictions, probabilities))]
    else:
        prediction_result = [{"Row Index": idx + 1, "Churn Prediction": 1-pred, "Churn Probability": 1-prob} 
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
