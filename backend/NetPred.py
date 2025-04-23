import model_building.NNet as NN
import os
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use('Agg')  # For headless environments (like servers)

directory = './backend/userfiles/'  # Path to uploaded Excel files

# Load pre-trained MLP model
Main_Model = NN.Churn_Network(init_mode="load_model", args="./backend/model_building/MLPCModel")

def predict_churn(file, sheet):
    file_path = os.path.join(directory, file)
    predictions = Main_Model.Sheet_Predict_default(file_path, sheet)

    # For churn probabilities (if available)
    if hasattr(Main_Model.neural_net, "predict_proba"):
        churn_probabilities = Main_Model.neural_net.predict_proba(Main_Model.X)[:, 1]
    else:
        churn_probabilities = [0.0] * len(predictions)

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

    return {"predictions": prediction_result}


def evaluate_model(file, sheet):
    file_path = os.path.join(directory, file)
    Main_Model.Sheet_Predict_default(file_path, sheet)

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

def get_features(file, sheet):
    file_path = os.path.join(directory, file)

    # Use your existing model data loader and preprocessor
    Main_Model.Sheet_Predict_default(file_path, sheet)

    # Run permutation importance on the fitted model
    result = permutation_importance(
        Main_Model.neural_net, 
        Main_Model.X, 
        Main_Model.Y, 
        n_repeats=10, 
        random_state=42, 
        scoring='accuracy'
    )

    print("results: ", result)

    # Convert to DataFrame for easier sorting and formatting
    importance_df = pd.DataFrame({
        'Feature': Main_Model.X.columns,
        'Importance': result.importances_mean
    })

    # Sort descending by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)

    feature_importances = importance_df.to_dict(orient='records')

    print("Feature importances: ", feature_importances)

    # Return keys with capital letters to match your frontend usage
    return {"features": feature_importances}
