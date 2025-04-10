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
models = joblib.load("./backend/model_building/ensemble_model.joblib")
xgb_model1 = models['xgb_model_1']
xgb_model2 = models['xgb_model_2']
ensemble_model = models['ensemble_model']

# Function to predict churn on new data using the ensemble model
def predict_churn(file, sheet):
    """Predict churn using the ensemble model"""
    # Load and preprocess data
    df = xgb.load_data(file, sheet)
    df_copy = df.copy()
    df = xgb.preprocess_data(df)
    
    # Prepare features
    X_unknown = df.drop(columns=['Churn'], errors='ignore')
    
    # Prepare data for BOTH base models (required for ensemble)
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
    
    # Make predictions using ensemble approach
    prob1 = xgb_model1.predict_proba(X1)[:, 1]
    prob2 = xgb_model2.predict_proba(X2)[:, 1]

    # Create a DataFrame for base model predictions
    base_predictions = np.vstack([prob1, prob2]).T 
    # Get meta-model weights from the ensemble model (final estimator of the ensemble)
    meta_weights = ensemble_model.final_estimator_.coef_[0]
    # Normalize meta-model weights (optional)
    norm_weights = meta_weights / np.sum(np.abs(meta_weights))

    # Calculate the combined probability using the meta weights
    probabilities = np.dot(base_predictions, norm_weights)

    # Make final predictions (thresholded at 0.5)
    predictions = (probabilities >= 0.5).astype(int)
    
    # Format results
    if 'Device number' in df_copy.columns:
        device_numbers = df_copy['Device number'].tolist()
        prediction_result = [
            {
                "Row Index": idx + 1,
                "Device number": device,
                "Churn Prediction": int(pred),
                "Churn Probability": float(prob),
            }
            for idx, (device, pred, prob) in enumerate(zip(device_numbers, predictions, probabilities))
        ]
    else:
        prediction_result = [
            {
                "Row Index": idx + 1,
                "Churn Prediction": int(pred),
                "Churn Probability": float(prob),
            }
            for idx, (pred, prob) in enumerate(zip(predictions, probabilities))
        ]

    return {"predictions": prediction_result}

def get_features(file, sheet):
    """Get combined feature importances from the ensemble model"""
    # Load and preprocess data (just to get feature names)
    df = xgb.load_data(file, sheet)
    df = xgb.preprocess_data(df)

    # Get base model importances
    importance_1 = xgb_model1.feature_importances_
    importance_2 = xgb_model2.feature_importances_
    
    # Get meta-model weights (how much each base model contributes)
    meta_weights = ensemble_model.final_estimator_.coef_[0]
    
    # Normalize weights to sum to 1
    norm_weights = meta_weights / np.sum(np.abs(meta_weights))
    
    # Create combined importance dictionary
    combined_importance = {}
    
    # Add Model 1's features (weighted)
    for feature, imp in zip(xgb_model1.feature_names_in_, importance_1):
        combined_importance[feature] = imp * norm_weights[0]
    
    # Add Model 2's features (weighted)
    for feature, imp in zip(xgb_model2.feature_names_in_, importance_2):
        if feature in combined_importance:
            combined_importance[feature] += imp * norm_weights[1]
        else:
            combined_importance[feature] = imp * norm_weights[1]
    
    # Convert to DataFrame and sort
    combined_importance = pd.DataFrame({
        'Feature': list(combined_importance.keys()),
        'Importance': list(combined_importance.values())
    }).sort_values('Importance', ascending=False)
    
    # Filter to only include features present in the current dataframe
    df_columns = set(df.columns)
    filtered_importance = combined_importance[combined_importance['Feature'].isin(df_columns)]
    
    return {"features": filtered_importance.to_dict(orient="records")}

def evaluate_model(file, sheet):
    """Evaluate using the ensemble model"""
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
    
    # Prepare data for ensemble model
    valid_indices = df.dropna(subset=['Churn']).index
    X_unknown = X_unknown.loc[valid_indices]
    
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
    
    # Make predictions using ensemble approach
    prob1 = xgb_model1.predict_proba(X1)[:, 1]
    prob2 = xgb_model2.predict_proba(X2)[:, 1]

    # Create a DataFrame for base model predictions
    base_predictions = np.vstack([prob1, prob2]).T 

    # Get meta-model weights from the ensemble model (final estimator of the ensemble)
    meta_weights = ensemble_model.final_estimator_.coef_[0]
    # Normalize meta-model weights (optional)
    norm_weights = meta_weights / np.sum(np.abs(meta_weights))

    # Calculate the combined probability using the meta weights
    probabilities = np.dot(base_predictions, norm_weights)

    # Make final predictions (thresholded at 0.5)
    predictions = (probabilities >= 0.5).astype(int)
    
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
    plt.title('Confusion Matrix (ensemble)')
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