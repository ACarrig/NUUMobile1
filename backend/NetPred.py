import model_building.NNet as NN
import os
from pathlib import Path
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
import shap
import warnings
from sklearn.impute import SimpleImputer
import warnings
import matplotlib
matplotlib.use('Agg')  # For headless environments (like servers)

# Get the absolute path to the directory where this script resides
BASE_DIR = Path(__file__).resolve().parent.parent  # Adjust how many levels up you want

# User files directory (uploaded Excel files)
directory = BASE_DIR / 'backend' / 'userfiles'

# Convert to string if your code expects str paths
directory = str(directory)

# Load pre-trained MLP model
model_path = BASE_DIR / 'backend' / 'model_building' / 'MLPCModel'
model_path = str(model_path)

Main_Model = NN.Churn_Network(init_mode="load_model", args=model_path)

def predict_churn(file, sheet):
    warnings.filterwarnings("ignore")
    file_path = os.path.join(directory, file)

    predictions = Main_Model.Sheet_Predict_default(file_path, sheet)

    # Impute missing values before prediction
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(Main_Model.X)

    # For churn probabilities (if available)
    churn_probabilities = (
        Main_Model.neural_net.predict_proba(X_imputed)[:, 1]
        if hasattr(Main_Model.neural_net, "predict_proba")
        else [0.0] * len(X_imputed)
    )

    df = pd.read_excel(file_path, sheet_name=sheet)
    if 'Device number' in df.columns:
        device_numbers = df['Device number'].tolist()
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
        warnings.resetwarnings()
    return {"predictions": prediction_result}

def download_churn(file, sheet):
    file_path = os.path.join(directory, file)
    
    # Load the original dataframe (unprocessed)
    df = pd.read_excel(file_path, sheet_name=sheet)
    df_copy = df.copy()

    # Use your model's preprocess and predict methods
    Main_Model.Sheet_Predict_default(file_path, sheet)

    # Impute missing values before prediction
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(Main_Model.X)

    # Predict churn probabilities and predictions
    if hasattr(Main_Model.neural_net, "predict_proba"):
        churn_probabilities = Main_Model.neural_net.predict_proba(Main_Model.X)[:, 1]
    else:
        churn_probabilities = [0.0] * len(Main_Model.X)

    predictions = Main_Model.neural_net.predict(Main_Model.X)

    # Append churn prediction and probabilities to original dataframe copy
    df_copy['Churn Probability'] = churn_probabilities
    df_copy['Churn Prediction'] = predictions

    # Reorder columns to put 'Churn' next to 'Churn Probability', if 'Churn' exists
    cols = df_copy.columns.tolist()
    if 'Churn' in cols and 'Churn Probability' in cols:
        cols.remove('Churn')
        insert_idx = cols.index('Churn Probability') + 1
        cols.insert(insert_idx, 'Churn')
        df_copy = df_copy[cols]

    prediction_result = df_copy.to_dict(orient='records')

    return {"predictions": prediction_result}

def get_features(file, sheet):
    file_path = os.path.join(directory, file)

    # Load and preprocess data using your existing code
    Main_Model.Sheet_Predict_default(file_path, sheet)

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(Main_Model.X)
    feature_names = Main_Model.X.columns.tolist()

    # Try SHAP for feature importance
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Use a sample of the data for background to speed up SHAP
            background = X_imputed[np.random.choice(X_imputed.shape[0], min(10, X_imputed.shape[0]), replace=False)]

            # Define prediction function for SHAP
            def model_predict_proba_pos(data):
                return Main_Model.neural_net.predict_proba(data)[:, 1]

            explainer = shap.KernelExplainer(model_predict_proba_pos, background)

            sample_limit = min(50, X_imputed.shape[0])
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

    except Exception as e:
        print(f"SHAP failed: {e}, falling back to permutation importance...")

        # Fallback: permutation importance
        try:
            result = permutation_importance(
                Main_Model.neural_net,
                X_imputed,
                Main_Model.Y,
                n_repeats=10,
                random_state=42,
                scoring='accuracy'
            )

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': result.importances_mean
            }).sort_values('Importance', ascending=False)

            return {
                "features": importance_df.to_dict(orient="records"),
                "method": "Permutation importance (fallback)"
            }

        except Exception as e2:
            print(f"Permutation importance also failed: {e2}")
            return {
                "features": [],
                "method": "No feature importance available"
            }

def evaluate_model(file, sheet):
    file_path = os.path.join(directory, file)

    Main_Model.Sheet_Predict_default(file_path, sheet)

    # Impute missing values before prediction
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(Main_Model.X)

    y_true = Main_Model.Y
    y_pred = Main_Model.predict(Main_Model.X)

    print(f"y_true: {y_true}, y_pred: {y_pred}")

    # Filter out rows where y_true == -1
    mask = y_true != -1
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # Metrics
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    precision = precision_score(y_true_filtered, y_pred_filtered, zero_division=0)
    recall = recall_score(y_true_filtered, y_pred_filtered, zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_filtered, zero_division=0)

    print(f"Accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1_score}")

    # Confusion matrix image
    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
    print("Confusion matrix:\n", cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    cm_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix_image": cm_base64,
    }
