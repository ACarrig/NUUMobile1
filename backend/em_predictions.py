import os
import pandas as pd
import numpy as np
import json
import shap
import io, base64
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import model_building.em_model as em
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer

import matplotlib
matplotlib.use('Agg')

# Load the ensemble model and metadata
model_data = joblib.load("./backend/model_building/ensemble_model.joblib")
ensemble_model = model_data['ensemble']
imputer = model_data['imputer']
threshold = model_data['best_threshold']
feature_names = model_data['feature_names']

directory = './backend/userfiles/'  # Path to user files folder

def make_predictions(df):
    # Use the ensemble model's preprocessing
    df = em.preprocess_data(df)
    X = df.drop(columns=['Churn'], errors='ignore')
    
    # One-hot encode and align with training features
    X_encoded = pd.get_dummies(X, drop_first=True)
    for col in set(feature_names) - set(X_encoded.columns):
        X_encoded[col] = 0
    X_encoded = X_encoded[feature_names]
    
    # Fill missing values using saved imputer
    X_imputed = imputer.transform(X_encoded)
    
    probs = ensemble_model.predict_proba(X_imputed)[:, 1]
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
    """Get combined feature importances from the ensemble model"""
    import os

    file_path = os.path.join(directory, file)
    df = pd.read_excel(file_path, sheet)
    df = em.preprocess_data(df)

    importance_data = []

    # Extract base models from saved ensemble
    if 'ensemble' in model_data:
        ensemble = model_data['ensemble']

        # Access base estimators depending on model type
        if hasattr(ensemble, 'estimators_'):  # StackingClassifier attribute after fit
            base_models = dict(ensemble.named_estimators_)
        elif hasattr(ensemble, 'estimators'):  # VotingClassifier attribute
            base_models = dict(ensemble.named_estimators)
        else:
            base_models = {}

        for name, model in base_models.items():
            # If model is a pipeline, unwrap final estimator
            if hasattr(model, 'named_steps'):
                final_model = model.named_steps[next(reversed(model.named_steps))]
            else:
                final_model = model

            # If calibrated classifier, get base_estimator_
            if hasattr(final_model, 'base_estimator_'):
                final_model = final_model.base_estimator_

            # Now get feature importances or coef_
            if hasattr(final_model, 'feature_importances_'):
                importances = final_model.feature_importances_
            elif hasattr(final_model, 'coef_'):
                importances = np.abs(final_model.coef_[0])
            else:
                # Cannot extract importances from this model
                importances = None

            if importances is not None:
                importance_data.extend(zip(feature_names, importances, [name] * len(feature_names)))

    if importance_data:
        importance_df = pd.DataFrame(importance_data, columns=['Feature', 'Importance', 'Model'])
        importance_df['Importance'] = importance_df.groupby('Model')['Importance'].transform(lambda x: x / x.sum())
        aggregated_importance = importance_df.groupby('Feature')['Importance'].mean().reset_index()
        aggregated_importance = aggregated_importance.sort_values('Importance', ascending=False)
        print("Feature importances:\n", aggregated_importance)
        return {"features": aggregated_importance.to_dict(orient="records")}
    else:
        # fallback with zeros for features present in data
        return {"features": [{"Feature": f, "Importance": 0} for f in feature_names if f in df.columns]}

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
    plt.title('Confusion Matrix (Ensemble Model)')
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