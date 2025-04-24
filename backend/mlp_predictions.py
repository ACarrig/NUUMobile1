import os
import joblib
import pandas as pd
import numpy as np
import json
import shap
import warnings
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

directory = './backend/userfiles/'  # Path to user files folder

def make_predictions(df):
    model_data = joblib.load("./backend/model_building/mlp_model.joblib")
    model = model_data['model']
    feature_names = model_data['feature_names']

    X_unknown = df.drop(columns=['Churn'], errors='ignore')

    missing_cols = set(feature_names) - set(X_unknown.columns)
    if missing_cols:
        print(f"Missing columns detected: {missing_cols}, retraining model...")
        model, feature_names = mlp.retrain_model(df)
        print("Retraining complete, updated model and features loaded.")

    # Reorder columns to match feature_names exactly
    X = X_unknown.loc[:, feature_names]

    # Impute missing values before prediction
    imputer = SimpleImputer(strategy='mean')  # or 'median', or 'most_frequent'
    X_imputed = imputer.fit_transform(X)

    predictions = model.predict(X_imputed)
    probabilities = model.predict_proba(X_imputed)
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
    file_path = os.path.join(directory, file)
    df = mlp.load_data(file_path, sheet)
    df_copy = df.copy()
    df = mlp.preprocess_data(df)

    probabilities, predictions = make_predictions(df)

    df_copy['Churn Probability'] = probabilities[:, 1]
    df_copy['Churn Prediction'] = predictions

    cols = df_copy.columns.tolist()
    if 'Churn' in cols:
        cols.remove('Churn')
        insert_index = cols.index('Churn Probability')
        cols.insert(insert_index, 'Churn')
        df_copy = df_copy[cols]

    prediction_result = df_copy.to_dict(orient='records')

    return {"predictions": prediction_result}

def get_features(file, sheet):
    try:
        file_path = os.path.join(directory, file)
        df = mlp.load_data(file_path, sheet)
        df = mlp.preprocess_data(df)

        model_data = joblib.load("./backend/model_building/mlp_model.joblib")
        model = model_data['model']
        feature_names = model_data['feature_names']

        missing_cols = set(feature_names) - set(df.columns)
        if missing_cols:
            print(f"Missing columns: {missing_cols}, retraining model...")
            model, feature_names = mlp.retrain_model(df)

        imputer = SimpleImputer(strategy='median')
        X = df[feature_names]
        X_imputed = imputer.fit_transform(X)

        # Predict on data (needed for SHAP explanation)
        probabilities, predictions = make_predictions(df)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                background = X_imputed[np.random.choice(X_imputed.shape[0], min(50, X_imputed.shape[0]), replace=False)]

                def model_predict_proba_pos(data):
                    return model.predict_proba(data)[:, 1]

                explainer = shap.KernelExplainer(model_predict_proba_pos, background)

                sample_limit = min(100, X_imputed.shape[0])
                shap_values = explainer.shap_values(X_imputed[:sample_limit])

                mean_abs_shap = np.abs(shap_values).mean(axis=0)

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": mean_abs_shap
            }).sort_values(by="Importance", ascending=False)

            return {
                "features": importance_df.to_dict(orient="records"),
                "method": "SHAP (prediction based)"
            }

        except Exception as shap_err:
            print(f"SHAP failed: {shap_err}, falling back to weight-based importance...")

            base_model = getattr(model, 'calibrated_classifiers_', [None])[0]
            if base_model and hasattr(base_model, 'estimator'):
                base_model = base_model.estimator
            elif hasattr(model, 'base_estimator'):
                base_model = model.base_estimator
            else:
                base_model = model

            if not hasattr(base_model, 'coefs_'):
                raise ValueError("Model does not contain coefs_ — not an MLPClassifier.")

            input_weights = base_model.coefs_[0]
            importance_scores = np.mean(np.abs(input_weights), axis=1)

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance_scores
            }).sort_values(by="Importance", ascending=False)

            return {
                "features": importance_df.to_dict(orient="records"),
                "method": "MLP Weight-Based"
            }

    except Exception as e:
        print(f"Error getting feature importances: {e}")
        return {"error": f"Could not get feature importances: {e}"}

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
