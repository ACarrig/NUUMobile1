import model_building.NNet as NN
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')

Main_Model = NN.Churn_Network(init_mode="load_model", args="model_building/MLPCModel")

def predict_churn(file, sheet):
    predictions = Main_Model.Sheet_Predict_default(file, sheet)
    # Format results
    if 'Device number' in df.columns:
        device_numbers = df['Device number'].tolist()
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
    # get the feature importances of the model

    # Filter to only include features present in current dataframe
    df_columns = set(df.columns)
    filtered_importance = feature_importance[feature_importance['Feature'].isin(df_columns)]

    return {"features": }

def evaluate_model(file, sheet):

    # Confusion Matrix
    cm = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix (ensemble)')
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