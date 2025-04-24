import os
import joblib
import pandas as pd
import numpy as np
import json
import io
import base64
from datetime import datetime
import backend.model_building.em_model2 as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')

directory = './backend/userfiles/'

# Load the ensemble model
MODEL_PATH = './backend/model_building/ensemble_model2.joblib'
model_data = joblib.load(MODEL_PATH)
ensemble_model = model_data['ensemble']
xgb_model1 = model_data['xgb_model_1']
xgb_model2 = model_data['xgb_model_2']

def select_model(X_unknown):
    """Select the best model based on feature overlap, with fallback to ensemble when overlaps are similar."""
    model1_features = set(xgb_model1.feature_names_in_)
    model2_features = set(xgb_model2.feature_names_in_)
    input_features = set(X_unknown.columns)
    
    # Calculate overlap with each model
    model1_overlap = len(model1_features.intersection(input_features))
    model2_overlap = len(model2_features.intersection(input_features))
    
    # Calculate overlap ratios
    model1_ratio = model1_overlap / len(model1_features) if len(model1_features) > 0 else 0
    model2_ratio = model2_overlap / len(model2_features) if len(model2_features) > 0 else 0
    
    # Determine if overlaps are similar (within 20% of each other)
    similarity_threshold = 0.2  # 20% difference threshold
    overlaps_are_similar = abs(model1_ratio - model2_ratio) <= similarity_threshold
    
    # Choose the model
    if overlaps_are_similar:
        return ensemble_model, "ensemble"
    elif model1_overlap >= model2_overlap:
        return xgb_model1, "model1"
    else:
        return xgb_model2, "model2"

def prepare_data_for_model(model, X_unknown):
    """Prepare data for the selected model (base or ensemble)."""
    if hasattr(model, 'feature_names_in_'):  # For base models
        X_prepared = X_unknown.loc[:, X_unknown.columns.isin(model.feature_names_in_)]
        missing_cols = set(model.feature_names_in_) - set(X_prepared.columns)
        for col in missing_cols:
            X_prepared[col] = 0
        X_prepared = X_prepared[model.feature_names_in_]
    else:  # For ensemble model
        # For ensemble, we need to prepare data for both base models
        X1 = X_unknown.loc[:, X_unknown.columns.isin(xgb_model1.feature_names_in_)]
        missing_cols1 = set(xgb_model1.feature_names_in_) - set(X1.columns)
        for col in missing_cols1:
            X1[col] = 0
        X1 = X1[xgb_model1.feature_names_in_]
        
        X2 = X_unknown.loc[:, X_unknown.columns.isin(xgb_model2.feature_names_in_)]
        missing_cols2 = set(xgb_model2.feature_names_in_) - set(X2.columns)
        for col in missing_cols2:
            X2[col] = 0
        X2 = X2[xgb_model2.feature_names_in_]
        
        X_prepared = (X1, X2)
    
    return X_prepared

def make_predictions(model, model_type, X_prepared):
    """Make predictions using the selected model."""
    if model_type == "ensemble":
        # For ensemble model, we need to predict with both base models first
        X1, X2 = X_prepared
        prob1 = xgb_model1.predict_proba(X1)[:, 1]
        prob2 = xgb_model2.predict_proba(X2)[:, 1]
        probabilities = (prob1 + prob2) / 2  # Average probabilities
        predictions = (probabilities >= 0.5).astype(int)
    else:
        # For base models
        probabilities = model.predict_proba(X_prepared)[:, 1]
        predictions = model.predict(X_prepared)
    
    return predictions, probabilities

# Function to predict churn on new data using ensemble model
def predict_churn(file, sheet):
    """Predict churn using the model whose features best match the input data."""
    # Load and preprocess data
    file_path = os.path.join(directory, file)
    df = xgb.load_data(file_path, sheet)
    df_copy = df.copy()
    df = xgb.preprocess_data(df)
    
    # Prepare features
    X_unknown = df.drop(columns=['Churn'], errors='ignore')
    
    # Select the best model
    selected_model, model_type = select_model(X_unknown)
    
    # Prepare data for selected model
    X_prepared = prepare_data_for_model(selected_model, X_unknown)
    
    # Make predictions
    predictions, probabilities = make_predictions(selected_model, model_type, X_prepared)
    
    # Format results
    if 'Device number' in df_copy.columns:
        device_numbers = df_copy['Device number'].tolist()
        prediction_result = [
            {
                "Row Index": idx + 1,
                "Device number": device,
                "Churn Prediction": int(pred),
                "Churn Probability": float(prob),
                "Model Used": model_type
            }
            for idx, (device, pred, prob) in enumerate(zip(device_numbers, predictions, probabilities))
        ]
    else:
        prediction_result = [
            {
                "Row Index": idx + 1,
                "Churn Prediction": int(pred),
                "Churn Probability": float(prob),
                "Model Used": model_type
            }
            for idx, (pred, prob) in enumerate(zip(predictions, probabilities))
        ]

    return {"predictions": prediction_result}

def get_features(file, sheet):
    """Get combined feature importances from the ensemble model"""
    # Load and preprocess data (just to get feature names)
    file_path = os.path.join(directory, file)
    df = xgb.load_data(file_path, sheet)
    df = xgb.preprocess_data(df)
    
    # Get combined feature importance from ensemble
    combined_importance = xgb.get_combined_feature_importance(
        ensemble_model,
        xgb_model1,
        xgb_model2,
        xgb_model1.feature_names_in_,
        xgb_model2.feature_names_in_
    )
    
    # Filter to only include features present in the current dataframe
    df_columns = set(df.columns)
    filtered_importance = combined_importance[combined_importance['Feature'].isin(df_columns)]
    
    return {"features": filtered_importance.to_dict(orient="records")}

def evaluate_model(file, sheet):
    """Evaluate using the model whose features best match the input data."""
    # Load and preprocess data
    file_path = os.path.join(directory, file)
    df = xgb.load_data(file_path, sheet)
    df = xgb.preprocess_data(df)
    
    # Check for true labels
    if 'Churn' not in df.columns:
        return {"error": "True labels (Churn) are not available in the dataset."}
    
    # Prepare features
    y_true = df['Churn'].dropna()
    X_unknown = df.drop(columns=['Churn'], errors='ignore')
    
    if len(y_true) == 0:
        return {"error": "No valid true labels available for evaluation."}
    
    # Select the best model
    selected_model, model_type = select_model(X_unknown)
    
    # Prepare data for selected model
    valid_indices = df.dropna(subset=['Churn']).index
    X_prepared = prepare_data_for_model(selected_model, X_unknown.loc[valid_indices])
    
    # Make predictions
    predictions, _ = make_predictions(selected_model, model_type, X_prepared)
    
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
    plt.title(f'Confusion Matrix ({model_type})')
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