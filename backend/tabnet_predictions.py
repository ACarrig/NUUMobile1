import os
import pandas as pd
import numpy as np
import json
import shap
import torch
from sklearn.inspection import permutation_importance
import io, base64
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import model_building.mlp_model as mlp

import matplotlib
matplotlib.use('Agg')

# Load the TabNet model and metadata
model_data = joblib.load("./backend/model_building/tabnet_model.joblib")
tabnet_model = model_data['tabnet_model']
label_encoders = model_data['label_encoders']
feature_names = model_data['feature_names']
categorical_columns = model_data['categorical_columns']
numeric_columns = model_data['numeric_columns']

threshold = 0.5  # Same threshold used in training
directory = './backend/userfiles/'  # Path to user files folder

def make_predictions(df):
    df = mlp.preprocess_data(df)
    X = df.drop(columns=['Churn'], errors='ignore')

    # Encode categoricals using saved label encoders
    for col in categorical_columns:
        if col in X.columns:
            le = label_encoders[col]
            X[col] = X[col].fillna('NA')
            X[col] = X[col].astype(str).apply(lambda x: x if x in le.classes_ else 'NA')
            le_classes = np.array(le.classes_.tolist() + ['NA']) if 'NA' not in le.classes_ else le.classes_
            le.classes_ = le_classes
            X[col] = le.transform(X[col])
        else:
            X[col] = 0  # if the column is missing, fill with dummy value

    # Fill missing numeric values with median
    for col in numeric_columns:
        if col in X.columns:
            median_val = model_data['feature_names'].index(col)  # get median index
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = 0

    # Align columns
    missing_cols = set(feature_names) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[feature_names]  # Reorder to match training

    X_np = X.values
    probs = tabnet_model.predict_proba(X_np)[:, 1]
    preds = (probs > threshold).astype(int)

    return probs, preds

def predict_churn(file, sheet):
    file_path = os.path.join(directory, file)
    df = pd.read_excel(file_path, sheet)
    df_copy = df.copy()
    df = mlp.preprocess_data(df)
    
    probabilities, predictions = make_predictions(df)
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
    df = mlp.preprocess_data(df)

    probabilities, predictions = make_predictions(df)

    df_copy['Churn Probability'] = probabilities
    df_copy['Churn Prediction'] = predictions

    cols = df_copy.columns.tolist()
    if 'Churn' in cols:
        cols.remove('Churn')
        insert_index = cols.index('Churn Probability')
        cols.insert(insert_index, 'Churn')
        df_copy = df_copy[cols]

    prediction_result = df_copy.to_dict(orient='records')

    return {"predictions": prediction_result}

def get_features(file, sheet, sample_size=100, background_size=10):
    try:
        file_path = os.path.join(directory, file)
        df = pd.read_excel(file_path, sheet)
        df = mlp.preprocess_data(df)
        
        # Handle target variable creation/validation
        y_true = None
        if 'Churn' in df.columns:
            df = df.dropna(subset=['Churn'])  # Drop rows where Churn is NA
            y_true = df['Churn'].astype(int).values
        elif 'Type' in df.columns:
            # Create Churn column from Type if available
            df['Churn'] = np.where(df['Type'] == 'Return', 1, 
                                 np.where(df['Type'] == 'Repair', 0, np.nan))
            df = df.dropna(subset=['Churn'])
            y_true = df['Churn'].astype(int).values
        
        X_pred = df.drop(columns=['Churn', 'Type'], errors='ignore')
        
        # Handle missing columns by filling with 0
        missing_cols = set(feature_names) - set(X_pred.columns)
        for col in missing_cols:
            X_pred[col] = 0
        
        # Reorder columns to match feature_names
        X_pred = X_pred[feature_names]
        
        # Encode categorical features
        for col in categorical_columns:
            if col in X_pred.columns:
                le = label_encoders[col]
                X_pred[col] = X_pred[col].fillna('NA').astype(str)
                # Handle unseen categories by mapping to 'NA'
                X_pred[col] = X_pred[col].apply(lambda x: x if x in le.classes_ else 'NA')
                # Ensure 'NA' is in the encoder classes
                if 'NA' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'NA')
                X_pred[col] = le.transform(X_pred[col])
            else:
                X_pred[col] = 0  # if column is missing
        
        # Fill missing numeric columns
        for col in numeric_columns:
            if col in X_pred.columns:
                median_val = X_pred[col].median()
                X_pred[col] = X_pred[col].fillna(median_val)
            else:
                X_pred[col] = 0
        
        X_pred_np = X_pred.values
        
        n_samples = X_pred_np.shape[0]
        if n_samples == 0:
            return {"error": "No valid samples available after preprocessing"}
        
        # Sample indices for evaluation
        if n_samples > sample_size:
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        else:
            sample_indices = np.arange(n_samples)
        
        X_sample = X_pred_np[sample_indices]
        y_sample = y_true[sample_indices] if y_true is not None else None
        
        # Background data for SHAP
        if n_samples > background_size:
            background_indices = np.random.choice(n_samples, background_size, replace=False)
        else:
            background_indices = np.arange(n_samples)
        background = X_pred_np[background_indices]
        
        # Try SHAP first
        try:
            print("Running SHAP KernelExplainer...")
            explainer = shap.KernelExplainer(tabnet_model.predict, background)
            shap_values = explainer.shap_values(X_sample, nsamples=100)
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": mean_abs_shap
            }).sort_values("Importance", ascending=False)
            
            print("SHAP calculation succeeded.")
            return {
                "features": importance_df.to_dict(orient="records"),
                "method": "SHAP KernelExplainer (sampled)"
            }
        
        except Exception as shap_e:
            print(f"SHAP calculation failed: {shap_e}")

            # Fallback to TabNet's built-in feature importances
            try:
                print("Falling back to TabNet weight-based feature importances...")
                
                # Get feature importances from the trained TabNet model
                if hasattr(tabnet_model, 'feature_importances_'):
                    feature_importances = tabnet_model.feature_importances_
                else:
                    # For some TabNet versions, we need to get it differently
                    feature_importances = np.mean(tabnet_model.feature_importances(X_pred_np), axis=0)
                
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": feature_importances
                }).sort_values("Importance", ascending=False)
                
                print("Weight-based feature importances calculated successfully.")
                return {
                    "features": importance_df.to_dict(orient="records"),
                    "method": "TabNet Weight-based Importance (final fallback)",
                    "warning": "Using model-internal weights which may be less reliable than SHAP or permutation importance"
                }
            
            except Exception as weight_e:
                print(f"Weight-based importance fallback failed: {weight_e}")
                return {
                    "error": "All feature importance methods failed",
                    "details": {
                        "shap_error": str(shap_e),
                        "weight_error": str(weight_e) if 'weight_e' in locals() else "Not attempted"
                    }
                }
    except Exception as e:
        print(f"Error getting feature importances: {e}")
        return {"error": f"Could not get feature importances: {str(e)}"}

def evaluate_model(file, sheet):
    file_path = os.path.join(directory, file)
    df = pd.read_excel(file_path, sheet)
    df_copy = df.copy()
    df = mlp.preprocess_data(df)

    if 'Churn' not in df_copy.columns and 'Type' in df_copy.columns:
        df['Churn'] = np.where(df_copy['Type'] == 'Return', 1, np.where(df_copy['Type'] == 'Repair', 0, np.nan))
    elif 'Churn' not in df_copy.columns:
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
    plt.title('Confusion Matrix (MLP)')
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
