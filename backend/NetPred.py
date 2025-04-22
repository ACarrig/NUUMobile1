import os
import joblib
import pandas as pd
import numpy as np
import json
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import model_building.mlp_model as mlp

import matplotlib
matplotlib.use('Agg')

# Load the MLP model
model_data = joblib.load("./backend/model_building/mlp_model.joblib")
mlp_model = model_data['model']
feature_names = model_data['feature_names']

directory = './backend/userfiles/'

def make_predictions(df):
    """Make predictions using the MLP model with proper data handling"""
    try:
        # Prepare features - create a copy to avoid SettingWithCopyWarning
        X_unknown = df.drop(columns=['Churn'], errors='ignore').copy()
        
        # Initialize a DataFrame with the correct feature columns
        X = pd.DataFrame(columns=feature_names)
        
        # Fill in available features
        for col in feature_names:
            if col in X_unknown.columns:
                X[col] = X_unknown[col]
            else:
                # For missing features, use median of available features
                X[col] = np.median(X_unknown.select_dtypes(include=np.number).values) if len(X_unknown) > 0 else 0
        
        # Separate numeric and non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]

        # Impute only numeric columns
        imputer = SimpleImputer(strategy='median')
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

        # Final feature set (DataFrame) with proper column names
        X_final = X[feature_names] if set(feature_names).issubset(X.columns) else X

        # Make predictions
        predictions = mlp_model.predict(X_final)
        probabilities = mlp_model.predict_proba(X_final)

        return probabilities, predictions
    
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

def predict_churn(file, sheet):
    """Predict churn using the MLP model"""
    try:
        # Load and preprocess data
        file_path = os.path.join(directory, file)
        df = pd.read_excel(file_path, sheet_name=sheet)
        df_copy = df.copy()
        
        # Preprocess the data
        df = mlp.preprocess_data(df)
        
        if len(df) == 0:
            return {"error": "No valid data available for prediction"}
        
        probabilities, predictions = make_predictions(df)
        churn_probabilities = probabilities[:, 1]
        
        # Format results
        result_key = "Device number" if 'Device number' in df_copy.columns else "Row Index"
        device_numbers = df_copy['Device number'].tolist() if 'Device number' in df_copy.columns else range(1, len(df)+1)
        
        prediction_result = [
            {
                "Row Index": idx + 1,
                result_key: device,
                "Churn Prediction": int(pred),
                "Churn Probability": float(prob),
            }
            for idx, (device, pred, prob) in enumerate(zip(device_numbers, predictions, churn_probabilities))
        ]

        return {"predictions": prediction_result}
    
    except Exception as e:
        return {"error": str(e)}

def download_churn(file, sheet):
    """Predict churn and return full dataframe"""
    try:
        file_path = os.path.join(directory, file)
        df = pd.read_excel(file_path, sheet_name=sheet)
        df_copy = df.copy()
        
        df = mlp.preprocess_data(df)
        probabilities, predictions = make_predictions(df)
        
        # Add predictions to original dataframe
        df_copy['Churn Probability'] = probabilities[:, 1]
        df_copy['Churn Prediction'] = predictions
        
        # Reorder columns if Churn exists
        if 'Churn' in df_copy.columns:
            cols = df_copy.columns.tolist()
            cols.remove('Churn')
            insert_index = cols.index('Churn Probability') if 'Churn Probability' in cols else 0
            cols.insert(insert_index, 'Churn')
            df_copy = df_copy[cols]
        
        return {"predictions": df_copy.to_dict(orient='records')}
    
    except Exception as e:
        return {"error": str(e)}

def get_features(file, sheet):
    """Calculate feature importances using permutation importance"""
    try:
        file_path = os.path.join(directory, file)
        df = pd.read_excel(file_path, sheet_name=sheet)
        df = mlp.preprocess_data(df)
        
        if 'Churn' not in df.columns:
            return {"features": [{"Feature": f, "Importance": 0} for f in feature_names]}
        
        # Prepare data
        X = df.drop(columns=['Churn'])
        y = df['Churn'].values
        
        # Get predictions with proper feature handling
        probabilities, _ = make_predictions(df)
        
        # Calculate permutation importance
        result = permutation_importance(
            mlp_model, 
            X.values if hasattr(X, 'values') else X, 
            y, 
            n_repeats=5, 
            random_state=42,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': result.importances_mean
        }).sort_values('Importance', ascending=False)
        
        return {"features": importance_df.to_dict(orient="records")}
    
    except Exception as e:
        return {"error": str(e), "features": [{"Feature": f, "Importance": 0} for f in feature_names]}

def evaluate_model(file, sheet):
    """Evaluate model performance"""
    try:
        file_path = os.path.join(directory, file)
        df = pd.read_excel(file_path, sheet_name=sheet)
        df_copy = df.copy()
        
        df = mlp.preprocess_data(df)
        
        # Handle Churn column
        if 'Churn' not in df.columns and 'Type' in df.columns:
            df['Churn'] = np.where(df_copy['Type'] == 'Return', 1, 
                                 np.where(df_copy['Type'] == 'Repair', 0, np.nan))
        
        if 'Churn' not in df.columns:
            return {"error": "No Churn information available for evaluation"}
        
        df_eval = df.dropna(subset=['Churn'])
        if len(df_eval) == 0:
            return {"error": "No valid Churn labels for evaluation"}
            
        y_true = df_eval['Churn'].astype(int).values
        probabilities, predictions = make_predictions(df_eval)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(y_true, predictions),
            "recall": recall_score(y_true, predictions),
            "f1_score": f1_score(y_true, predictions)
        }
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, predictions)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn', 
                    xticklabels=['No Churn', 'Churn'], 
                    yticklabels=['No Churn', 'Churn'])
        plt.title('MLP Model Performance')
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        metrics["confusion_matrix_image"] = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()
        
        return metrics
    
    except Exception as e:
        return {"error": str(e)}