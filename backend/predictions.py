import os
import joblib
import pandas as pd
import numpy as np
import json
import io
import base64
from datetime import datetime
import model_building.xgb_model as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')

# Load the ensemble model
ENSEMBLE_MODEL_PATH = "./backend/model_building/ensemble_model.joblib"
XGB_MODEL1_PATH = "./backend/model_building/xgb_model1.joblib"
XGB_MODEL2_PATH = "./backend/model_building/xgb_model2.joblib"

if os.path.exists(ENSEMBLE_MODEL_PATH):
    ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)
    xgb_model1 = joblib.load(XGB_MODEL1_PATH)
    xgb_model2 = joblib.load(XGB_MODEL2_PATH)
else:
    raise FileNotFoundError("Model files not found!")

# Function to predict churn on new data using ensemble model
def predict_churn(file, sheet):
    """Predict churn using the model whose features best match the input data."""
    # Load and preprocess data
    df = xgb.load_data(file, sheet)
    original_df = df.copy()
    df = xgb.preprocess_data(df)
    
    # Prepare features
    X_unknown = df.drop(columns=['Churn'], errors='ignore')
    
    # Determine which base model's features match better
    model1_features = set(xgb_model1.feature_names_in_)
    model2_features = set(xgb_model2.feature_names_in_)
    input_features = set(X_unknown.columns)
    
    # Calculate overlap with each model
    model1_overlap = len(model1_features.intersection(input_features))
    model2_overlap = len(model2_features.intersection(input_features))
    
    # Choose the model with better feature match
    if model1_overlap >= model2_overlap:
        selected_model = xgb_model1
    else:
        selected_model = xgb_model2
    
    # Prepare data for selected model
    X_prepared = X_unknown.loc[:, X_unknown.columns.isin(selected_model.feature_names_in_)]
    missing_cols = set(selected_model.feature_names_in_) - set(X_prepared.columns)
    for col in missing_cols:
        X_prepared[col] = 0
    X_prepared = X_prepared[selected_model.feature_names_in_]
    
    # Make predictions
    probabilities = selected_model.predict_proba(X_prepared)[:, 1]
    predictions = selected_model.predict(X_prepared)
    
    # Format results
    if 'Device number' in original_df.columns:
        device_numbers = original_df['Device number'].tolist()
        prediction_result = [
            {
                "Row Index": idx + 1,
                "Device number": device,
                "Churn Prediction": int(pred),
                "Churn Probability": float(prob)
            }
            for idx, (device, pred, prob) in enumerate(zip(device_numbers, predictions, probabilities))
        ]
    else:
        prediction_result = [
            {
                "Row Index": idx + 1,
                "Churn Prediction": int(pred),
                "Churn Probability": float(prob)
            }
            for idx, (pred, prob) in enumerate(zip(predictions, probabilities))
        ]

    return {"predictions": prediction_result}

def get_features(file, sheet):
    """Get combined feature importances from the ensemble model"""
    # Load and preprocess data (just to get feature names)
    df = xgb.load_data(file, sheet)
    df = xgb.preprocess_data(df)
    
    # Get combined feature importance from ensemble
    combined_importance = xgb.get_combined_feature_importance(
        ensemble_model,
        xgb_model1,
        xgb_model2,
        xgb_model1.feature_names_in_,
        xgb_model2.feature_names_in_
    )
    
    return {"features": combined_importance.to_dict(orient="records")}

def evaluate_model(file, sheet):
    """Evaluate using the model whose features best match the input data."""
    # Load and preprocess data
    df = xgb.load_data(file, sheet)
    df = xgb.preprocess_data(df)
    
    # Check for true labels
    if 'Churn' not in df.columns:
        return {"error": "True labels (Churn) are not available in the dataset."}
    
    # Prepare features
    y_true = df['Churn'].dropna()
    X_unknown = df.drop(columns=['Churn'], errors='ignore')
    
    if len(y_true) == 0:
        return {"error": "No valid true labels available for evaluation."}
    
    # Determine which base model's features match better
    model1_features = set(xgb_model1.feature_names_in_)
    model2_features = set(xgb_model2.feature_names_in_)
    input_features = set(X_unknown.columns)
    
    model1_overlap = len(model1_features.intersection(input_features))
    model2_overlap = len(model2_features.intersection(input_features))
    
    if model1_overlap >= model2_overlap:
        selected_model = xgb_model1
    else:
        selected_model = xgb_model2
    
    # Prepare data for selected model
    valid_indices = df.dropna(subset=['Churn']).index
    X_prepared = X_unknown.loc[valid_indices]
    X_prepared = X_prepared.loc[:, X_prepared.columns.isin(selected_model.feature_names_in_)]
    missing_cols = set(selected_model.feature_names_in_) - set(X_prepared.columns)
    for col in missing_cols:
        X_prepared[col] = 0
    X_prepared = X_prepared[selected_model.feature_names_in_]
    
    # Make predictions
    predictions = selected_model.predict(X_prepared)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix_image": img_base64
    }
