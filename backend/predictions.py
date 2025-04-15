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
    
    # Handle NaN values
    X = X.fillna(X.median())

    # Robust NaN handling:
    # 1. First check if there are any NaN values
    if X.isnull().values.any():
        # 2. Calculate medians while ignoring NaN values
        medians = X.median(skipna=True)
        
        # 3. Handle case where entire column is NaN
        medians = medians.fillna(0)  # If median is NaN (all values were NaN), use 0
        
        # 4. Fill NaN values
        X = X.fillna(medians)
        
        # 5. Final check - if any NaN remains (shouldn't happen), fill with 0
        X = X.fillna(0)

    # Make predictions
    predictions = ensemble_model.predict(X)
    probabilities = ensemble_model.predict_proba(X)

    return probabilities, predictions

# Function to predict churn on new data using the ensemble model
def predict_churn(file, sheet):
    """Predict churn using the ensemble model"""
    # Load and preprocess data
    file_path = os.path.join(directory, file)
    df = xgb.load_data(file_path, sheet)
    df_copy = df.copy()
    df = xgb.preprocess_data(df)
    
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

def get_features(file, sheet):
    """Get combined feature importances from the ensemble model"""
    # Load and preprocess data
    file_path = os.path.join(directory, file)
    df = xgb.load_data(file_path, sheet)
    df = xgb.preprocess_data(df)

    # Get feature importances from the base models
    importance_data = []
    
    # Check if we have base models in our saved model data
    if 'base_models' in model_data:
        for name, model in model_data['base_models'].items():
            if hasattr(model, 'feature_importances_'):
                # For models with native feature importance (XGBoost, RandomForest)
                importances = model.feature_importances_
                importance_data.extend(zip(feature_names, importances, [name]*len(feature_names)))
            elif hasattr(model, 'coef_'):
                # For linear models (LogisticRegression)
                importances = np.abs(model.coef_[0])
                importance_data.extend(zip(feature_names, importances, [name]*len(feature_names)))

    # Create a DataFrame for visualization
    if importance_data:
        importance_df = pd.DataFrame(importance_data, columns=['Feature', 'Importance', 'Model'])
        
        # Normalize importance scores within each model
        importance_df['Importance'] = importance_df.groupby('Model')['Importance'].transform(
            lambda x: x / x.sum()
        )
        
        # Aggregate across models
        aggregated_importance = importance_df.groupby('Feature')['Importance'].mean().reset_index()
        aggregated_importance = aggregated_importance.sort_values('Importance', ascending=False)
        
        # Filter to only include features present in current dataframe
        df_columns = set(df.columns)
        filtered_importance = aggregated_importance[aggregated_importance['Feature'].isin(df_columns)]
        
        return {"features": filtered_importance.to_dict(orient="records")}
    else:
        # Fallback if no feature importances can be extracted
        return {"features": [{"Feature": f, "Importance": 0} for f in feature_names if f in df.columns]}
    
def evaluate_model(file, sheet):
    """Evaluate using the ensemble model"""
    # Load and preprocess data
    file_path = os.path.join(directory, file)
    df = xgb.load_data(file_path, sheet)
    df_copy = df.copy()

    df = xgb.preprocess_data(df)
    
    # Check if preprocessing created the 'Churn' column
    if 'Churn' not in df_copy.columns:
        # Check if we have the raw 'Type' column that could be used
        if 'Type' in df_copy.columns:
            # Manually create Churn column if preprocessing didn't
            df['Churn'] = np.where(df_copy['Type'] == 'Return', 1, 
                                 np.where(df_copy['Type'] == 'Repair', 0, np.nan))
            print("Manually created Churn column from Type")
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