import model_building.NNet as NN

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
    return {"features": }

def evaluate_model(file, sheet):

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix_image": img_base64
    }