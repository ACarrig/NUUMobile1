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
def preprocess_data(df):
    # Drop irrelevant columns
    columns_to_drop = ['Device number', 'Product/Model #', 'Office Date', 'Office Time In', 'Type', 'Final Status', 'Defect / Damage type', 'Responsible Party']
    df.drop(columns=columns_to_drop, inplace=True)

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

    # Define churn (1 if interval - activate > 30 days, else 0)
    df['Churn'] = (df['interval - activate'] > 30).astype(int)

    # Drop date columns after creating churn
    df.drop(columns=['interval_date', 'last_boot_date', 'active_date'], inplace=True)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df

# Function to handle missing values
def handle_missing_values(df):
    # Separate rows with missing values for later prediction
    unknown_data = df[df.isnull().any(axis=1)]
    # Drop rows with missing values for model prediction
    df_cleaned = df.dropna()
    return df_cleaned, unknown_data

def predict_churn(file, sheet):
    """Predict churn on new data from the specified Excel file and sheet."""
    # Load the file & sheet
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, sheet_name=sheet)

    # Store the device numbers for later use
    device_numbers = df['Device number'].copy()

    # Preprocess the data
    df = preprocess_data(df)

    # Handle missing values (drop rows with missing data for prediction)
    df_cleaned, _ = handle_missing_values(df)

    # Ensure that the columns in the input data match the columns used for training
    if 'Churn' in df_cleaned.columns:
        X_input = df_cleaned.drop(columns=['Churn'])
    else:
        X_input = df_cleaned

    # Align columns
    input_columns = X_input.columns
    model_columns = model.feature_names_in_

    # Add missing columns with default value 0
    missing_cols = set(model_columns) - set(input_columns)
    for col in missing_cols:
        X_input[col] = 0

    # Reorder the columns to match the model's expected order
    X_input = X_input[model_columns]

    # Make predictions using the trained model
    predictions = model.predict(X_input)

    # Convert predictions and device numbers into lists
    predictions = predictions.astype(int).tolist()
    device_numbers = device_numbers.tolist()

    # Generate a row index (1-based)
    prediction_result = [{"Row Index": idx + 1, "Device number": device, "Churn Prediction": pred} 
                         for idx, (device, pred) in enumerate(zip(device_numbers, predictions))]

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
