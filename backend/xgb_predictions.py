import os
import pandas as pd
import numpy as np
import json
import shap
import io, base64
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import model_building.xgb_model as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')

# Load the XGBoost model and metadata
model_data = joblib.load("./backend/model_building/xgb_model.joblib")
xgb_model = model_data['xgb']
feature_names = model_data['feature_names']
median_vals = model_data['median']

threshold = 0.4  # Same threshold as used during training

directory = './backend/userfiles/'  # Path to user files folder

def make_predictions(df):
    df = xgb.preprocess_data(df)
    X = df.drop(columns=['Churn'], errors='ignore')
    
    # One-hot encode and align with training features
    X_encoded = pd.get_dummies(X, drop_first=True)
    for col in set(feature_names) - set(X_encoded.columns):
        X_encoded[col] = 0
    X_encoded = X_encoded[feature_names]
    
    # Fill missing values using saved median
    X_encoded = X_encoded.fillna(median_vals)
    
    probs = xgb_model.predict_proba(X_encoded)[:, 1]
    preds = (probs > threshold).astype(int)
    
    return probs, preds

def predict_churn(file, sheet):
    file_path = os.path.join(directory, file)
    df = pd.read_excel(file_path, sheet)
    df_copy = df.copy()
    
    probabilities, predictions = make_predictions(df_copy)
    
    churn_probabilities = probabilities
    
    if 'Device number' in df_copy.columns:
        device_numbers = df_copy['Device number'].tolist()
        prediction_result = [
            {
                "Row Index": idx + 1,
                "Device number": device,
                "Churn Prediction": int(pred),
                "Churn Probability": float(prob),
            }
            for idx, (device, pred, prob) in enumerate(zip(device_numbers, predictions, churn_probabilities))
        ]
    else:
        prediction_result = [
            {
                "Row Index": idx + 1,
                "Churn Prediction": int(pred),
                "Churn Probability": float(prob),
            }
            for idx, (pred, prob) in enumerate(zip(predictions, churn_probabilities))
        ]
    
    return {"predictions": prediction_result}

def download_churn(file, sheet):
    file_path = os.path.join(directory, file)
    df = pd.read_excel(file_path, sheet)
    df_copy = df.copy()

    probabilities, predictions = make_predictions(df_copy)

    df_copy['Churn Probability'] = probabilities
    df_copy['Churn Prediction'] = predictions

    if 'Churn' in df_copy.columns:
        cols = df_copy.columns.tolist()
        cols.remove('Churn')
        insert_index = cols.index('Churn Probability')
        cols.insert(insert_index, 'Churn')
        df_copy = df_copy[cols]

    prediction_result = df_copy.to_dict(orient='records')
    return {"predictions": prediction_result}

def get_features(file, sheet):
    """Get feature importances using SHAP with proper data handling."""
    # Load and preprocess data
    file_path = os.path.join(directory, file)
    df = pd.read_excel(file_path, sheet)
    df = xgb.preprocess_data(df)
    
    # Convert all columns to numeric and clean data
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(df.median())
    
    # Prepare feature matrix
    X = df.loc[:, df.columns.isin(model_data['feature_names'])].copy()
    missing_cols = set(model_data['feature_names']) - set(X.columns)
    for col in missing_cols:
        X[col] = df[col].median() if col in df.columns else 0
    X = X[model_data['feature_names']]
    
    # Sample if large dataset
    X_sample = X.sample(n=min(200, len(X)), random_state=42) if len(X) > 200 else X.copy()
    X_sample = X_sample.astype(np.float32)  # Ensure proper dtype
    
    try:
        # Attempt SHAP explanation
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_importance = np.mean([np.abs(v).mean(axis=0) for v in shap_values], axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
            
    except Exception as e:
        print(f"[WARN] SHAP failed, using fallback: {e}")
        if hasattr(xgb_model, 'feature_importances_'):
            shap_importance = xgb_model.feature_importances_
        else:
            shap_importance = np.zeros(len(model_data['feature_names']))
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        "Feature": model_data['feature_names'],
        "Importance": shap_importance
    }).sort_values("Importance", ascending=False)

    return {"features": importance_df.to_dict(orient="records")}

def evaluate_model(file, sheet):
    """Evaluate using the XGBoost model"""
    file_path = os.path.join(directory, file)
    df = pd.read_excel(file_path, sheet)
    df_copy = df.copy()

    df = xgb.preprocess_data(df)
    
    if 'Churn' not in df_copy.columns:
        if 'Type' in df_copy.columns:
            df['Churn'] = np.where(df_copy['Type'] == 'Return', 1, 
                                   np.where(df_copy['Type'] == 'Repair', 0, np.nan))
        else:
            return {"error": "Dataset lacks both 'Churn' and 'Type' columns - Evaluation is not possible"}
        
    df_eval = df.dropna(subset=['Churn']).copy()
    y_true = df_eval['Churn'].astype(int).values
    
    probabilities, predictions = make_predictions(df_eval)
    
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    
    cm = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix (XGBoost)')
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
