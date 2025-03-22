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

    # Drop date columns after creating churn
    df.drop(columns=['interval_date', 'last_boot_date', 'active_date'], inplace=True)

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

    # Get the features for prediction
    X_unknown = df.drop(columns=['Churn'], errors='ignore')

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

    # Print only the 'Churn' and 'Churn_Predicted' columns
    print(original_df[['Churn', 'Churn_Predicted']].head())

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
