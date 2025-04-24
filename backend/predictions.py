import os
import joblib
import pandas as pd
import numpy as np
import json
import shap
import io
import base64
from datetime import datetime
import model_building.em_model as em
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')

# Load the ensemble model
model_data = joblib.load("./backend/model_building/ensemble_model.joblib")
ensemble_model = model_data['ensemble']
feature_names = model_data['feature_names']

directory = './backend/userfiles/'  # Path to user files folder

def make_predictions(df):
    # Prepare features
    X_unknown = df.drop(columns=['Churn'], errors='ignore')
    
    # Prepare data
    X = X_unknown.loc[:, X_unknown.columns.isin(feature_names)]
    missing_cols = set(feature_names) - set(X.columns)
    for col in missing_cols:
        X.loc[:, col] = X.median()
    X = X[feature_names]
    
    # Make predictions
    predictions = ensemble_model.predict(X)
    probabilities = ensemble_model.predict_proba(X)

    return probabilities, predictions

# Function to predict churn on new data using the ensemble model
def predict_churn(file, sheet):
    """Predict churn using the ensemble model"""
    # Load and preprocess data
    file_path = os.path.join(directory, file)
    df = em.load_data(file_path, sheet)
    df_copy = df.copy()
    df = em.preprocess_data(df)
    
    probabilities, predictions = make_predictions(df)
    
    # Extract the probability for class 1 (Churn)
    churn_probabilities = probabilities[:, 1]  # This gets the second column (index 1) for the positive class
    
    # Format results
    if 'Device number' in df_copy.columns:
        device_numbers = df_copy['Device number'].tolist()
        prediction_result = [
            {
                "Row Index": idx + 1,
                "Device number": device,
                "Churn Prediction": int(pred),
                "Churn Probability": float(prob),  # Now prob is a single float value
            }
            for idx, (device, pred, prob) in enumerate(zip(device_numbers, predictions, churn_probabilities))
        ]
    else:
        prediction_result = [
            {
                "Row Index": idx + 1,
                "Churn Prediction": int(pred),
                "Churn Probability": float(prob),  # Now prob is a single float value
            }
            for idx, (pred, prob) in enumerate(zip(predictions, churn_probabilities))
        ]

    return {"predictions": prediction_result}

def download_churn(file, sheet):
    """Predict churn using the ensemble model"""
    # Load and preprocess data
    file_path = os.path.join(directory, file)
    df = em.load_data(file_path, sheet)
    df_copy = df.copy()
    df = em.preprocess_data(df)

    probabilities, predictions = make_predictions(df)

    # Add predictions to original dataframe
    df_copy['Churn Probability'] = probabilities[:, 1]
    df_copy['Churn Prediction'] = predictions

    # Reorder columns: move 'Churn' next to 'Churn Probability'
    cols = df_copy.columns.tolist()
    if 'Churn' in cols:
        cols.remove('Churn')
        insert_index = cols.index('Churn Probability')
        cols.insert(insert_index, 'Churn')
        df_copy = df_copy[cols]

    # Convert to dictionary for JSON response
    prediction_result = df_copy.to_dict(orient='records')

    return {"predictions": prediction_result}

def get_features(file, sheet):
    """Get combined feature importances using SHAP with fallback to built-in importance scores."""
    # Load and preprocess data
    file_path = os.path.join(directory, file)
    df = em.load_data(file_path, sheet)
    df = em.preprocess_data(df)

    print("preprocessed df's columns: ", df.columns)
    
    # Ensure only features your model expects are selected
    X = df.loc[:, df.columns.isin(model_data['feature_names'])].copy()

    missing_cols = set(model_data['feature_names']) - set(X.columns)
    for col in missing_cols:
        if col in df.columns:
            X[col] = df[col].median()
        else:
            X[col] = 0  # or np.nan, or another fallback

    X = X[model_data['feature_names']]

    # Sample the data for SHAP to improve performance
    if len(X) > 200:
        X_sample = X.sample(n=200, random_state=42)
    else:
        X_sample = X.copy()

    importance_data = []

    if 'base_models' in model_data:
        for name, model in model_data['base_models'].items():
            try:
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)

                if shap_values.values.ndim == 3:
                    shap_importance = np.abs(shap_values.values).mean(axis=(0, 2))
                else:
                    shap_importance = np.abs(shap_values.values).mean(axis=0)

                importance_data.extend(zip(model_data['feature_names'], shap_importance, [name] * len(model_data['feature_names'])))

            except Exception as e:
                # Fallback to built-in feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    importance_data.extend(zip(model_data['feature_names'], importances, [name]*len(model_data['feature_names'])))
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_[0])
                    importance_data.extend(zip(model_data['feature_names'], importances, [name]*len(model_data['feature_names'])))
                else:
                    print(f"[WARN] No importances found for model: {name}. Error: {str(e)}")

    # Process the importance data
    if importance_data:
        importance_df = pd.DataFrame(importance_data, columns=['Feature', 'Importance', 'Model'])
        
        # Normalize within each model
        importance_df['Importance'] = importance_df.groupby('Model')['Importance'].transform(lambda x: x / x.sum())
        
        # Aggregate normalized importances across models
        aggregated = importance_df.groupby('Feature')['Importance'].mean().reset_index()
        aggregated = aggregated.sort_values('Importance', ascending=False)

        print("Feature importance: ", aggregated)
        return {"features": aggregated.to_dict(orient="records")}
    else:
        # fallback
        return {"features": [{"Feature": f, "Importance": 0} for f in model_data['feature_names']]}
    
def evaluate_model(file, sheet):
    """Evaluate using the ensemble model"""
    # Load and preprocess data
    file_path = os.path.join(directory, file)
    df = em.load_data(file_path, sheet)
    df_copy = df.copy()

    df = em.preprocess_data(df)
    
    # Check if preprocessing created the 'Churn' column
    if 'Churn' not in df_copy.columns:
        # Check if we have the raw 'Type' column that could be used
        if 'Type' in df_copy.columns:
            # Manually create Churn column if preprocessing didn't
            df['Churn'] = np.where(df_copy['Type'] == 'Return', 1, 
                                 np.where(df_copy['Type'] == 'Repair', 0, np.nan))
        else:
            return {"error": "Dataset lacks both 'Churn' and 'Type' columns - Evaluation is not possible"}
        
    # Only evaluate rows with known Churn
    df_eval = df.dropna(subset=['Churn']).copy()
    y_true = df_eval['Churn'].astype(int).values
    
    probabilities, predictions = make_predictions(df_eval)
    
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