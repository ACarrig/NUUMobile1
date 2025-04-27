import os
import pandas as pd
import numpy as np
import json
import shap
import io, base64
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import model_building.xgb_model as em
import model_building.em_model as ensemble_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')

# Load the ensemble model and metadata
model_data = joblib.load("./backend/model_building/ensemble_model.joblib")
ensemble = model_data['ensemble']
feature_names = model_data['feature_names']
median_vals = model_data['median']

threshold = 0.4  # Same threshold as used during training

directory = './backend/userfiles/'  # Path to user files folder

def make_predictions(df):
    df = em.preprocess_data(df)
    X = df.drop(columns=['Churn'], errors='ignore')
    
    # One-hot encode and align with training features
    X_encoded = pd.get_dummies(X, drop_first=True)
    for col in set(feature_names) - set(X_encoded.columns):
        X_encoded[col] = 0
    X_encoded = X_encoded[feature_names]
    
    # Fill missing values using saved median
    X_encoded = X_encoded.fillna(median_vals)
    
    probs = ensemble.predict_proba(X_encoded)[:, 1]
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

def retrain_model_with_new_data(df):
    # Preprocess your data
    df = em.preprocess_data(df)
    
    if 'Churn' not in df.columns:
        raise ValueError("Training data must include a 'Churn' column.")
    
    X = df.drop(columns=['Churn'])
    y = df['Churn'].astype(int)
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    median_vals = X_encoded.median()

    X_encoded = X_encoded.fillna(median_vals)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Retrain using the ensemble retraining function
    new_ensemble = ensemble_model.train_stacking_ensemble(X_train, y_train)

    # Save the updated model and its metadata
    joblib.dump({
        'ensemble': new_ensemble,
        'feature_names': X_encoded.columns.tolist(),
        'median': median_vals
    }, './backend/model_building/ensemble_model.joblib')

    print("Ensemble model retrained and saved with new features.")

def get_features(file, sheet):
    # Load and preprocess data
    file_path = os.path.join(directory, file)
    df = pd.read_excel(file_path, sheet)
    df = em.preprocess_data(df)

    # Select features
    X = df.loc[:, df.columns.isin(feature_names)].copy()

    # Add missing columns if any, filling with median or zero
    missing_cols = set(feature_names) - set(X.columns)
    for col in missing_cols:
        if col in df.columns:
            X[col] = df[col].median()
        else:
            X[col] = 0

    X = X[feature_names]

    # Convert all to numeric and fill NaNs
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    base_models = ensemble.named_estimators_

    importances = []

    # XGBoost importance
    xgb_model = base_models['xgb']
    if hasattr(xgb_model, 'feature_importances_'):
        xgb_imp = xgb_model.feature_importances_
    else:
        xgb_imp = np.zeros(len(feature_names))
    importances.append(('xgb', xgb_imp))

    # Random Forest importance
    rf_model = base_models['rf']
    if hasattr(rf_model, 'feature_importances_'):
        rf_imp = rf_model.feature_importances_
    else:
        rf_imp = np.zeros(len(feature_names))
    importances.append(('rf', rf_imp))

    # Logistic Regression importance: absolute normalized coefficients
    lr_model = base_models['lr']
    if hasattr(lr_model, 'coef_'):
        lr_coef = np.abs(lr_model.coef_).flatten()
        if lr_coef.sum() > 0:
            lr_imp = lr_coef / lr_coef.sum()
        else:
            lr_imp = np.zeros_like(lr_coef)
    else:
        lr_imp = np.zeros(len(feature_names))
    importances.append(('lr', lr_imp))

    # Average importances
    combined_importance = np.mean(np.vstack([imp for _, imp in importances]), axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": combined_importance
    }).sort_values("Importance", ascending=False)

    print("[INFO] Feature importances aggregated from ensemble base models (no SHAP).")

    return {"features": importance_df.to_dict(orient="records")}

def evaluate_model(file, sheet):
    """Evaluate using the ensemble model"""
    file_path = os.path.join(directory, file)
    df = pd.read_excel(file_path, sheet)
    df_copy = df.copy()

    df = em.preprocess_data(df)
    
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
    plt.title('Confusion Matrix (Ensemble)')
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
