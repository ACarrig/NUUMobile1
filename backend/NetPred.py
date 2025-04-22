import os
import joblib
import pandas as pd
import numpy as np
import json
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import model_building.mlp_model as mlp 

import matplotlib
matplotlib.use('Agg')

# Load the MLP model
model_data = joblib.load("./backend/model_building/mlp_model.joblib")
mlp_model = model_data['model']
feature_names = model_data['feature_names']

directory = './backend/userfiles/'  # Path to user files folder

def make_predictions(df):
    X_unknown = df.drop(columns=['Churn'], errors='ignore')
    
    # Prepare feature columns
    X = X_unknown.loc[:, X_unknown.columns.isin(feature_names)]
    
    # Impute missing columns (using median as default strategy)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Reorder columns based on feature names to match the model input
    X_imputed = X_imputed[feature_names]

    predictions = mlp_model.predict(X_imputed)
    probabilities = mlp_model.predict_proba(X_imputed)
    return probabilities, predictions

def predict_churn(file, sheet):
    file_path = os.path.join(directory, file)
    df = mlp.load_data(file_path, sheet)
    df_copy = df.copy()
    df = mlp.preprocess_data(df)
    
    probabilities, predictions = make_predictions(df)
    churn_probabilities = probabilities[:, 1]

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
    """Predict churn using the ensemble model"""
    # Load and preprocess data
    file_path = os.path.join(directory, file)
    df = mlp.load_data(file_path, sheet)
    df_copy = df.copy()
    df = mlp.preprocess_data(df)

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
    """ Get feature importance using Permutation Importance """
    file_path = os.path.join(directory, file)
    df = mlp.load_data(file_path, sheet)
    df = mlp.preprocess_data(df)

    # Drop rows where 'Churn' is NaN — required for evaluation
    df = df.dropna(subset=['Churn'])

    X = df[feature_names]  # Ensure this matches your model's features
    y = df['Churn'].astype(int)  # Convert to int just in case it's float

    # Handle missing values by imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    result = permutation_importance(
        mlp_model,
        X_imputed,
        y,
        n_repeats=10,
        random_state=42,
        scoring='roc_auc',
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    })

    importance_df = importance_df.sort_values('Importance', ascending=False)
    return {"features": importance_df.to_dict(orient="records")}

def evaluate_model(file, sheet):
    file_path = os.path.join(directory, file)
    df = mlp.load_data(file_path, sheet)
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
