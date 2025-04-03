import pandas as pd
import numpy as np
import json
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import timedelta

# Function to load dataset
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Function to preprocess SIM information
def classify_sim_info(sim_info):
    if isinstance(sim_info, str) and sim_info not in ['Unknown', ''] :
        try:
            parsed = json.loads(sim_info)
            if isinstance(parsed, list) and parsed:
                carrier_name = parsed[0].get('carrier_name', None)
                return 'inserted' if carrier_name and carrier_name != 'Unknown' else 'uninserted'
        except json.JSONDecodeError:
            return 'uninserted'
    return 'uninserted'

# Convert Arabic numerals to Western numerals
def convert_arabic_numbers(text):
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"
    return text.translate(str.maketrans(arabic_digits, western_digits)) if isinstance(text, str) else text

# Preprocessing function to clean and prepare the data
def preprocess_data(df):
    # Drop irrelevant columns
    columns_to_drop = ['Device number', 'Month', 'Office Date', 'Office Time In', 'Type', 'Final Status', 'Defect / Damage type', 'Responsible Party']
    df.drop(columns=columns_to_drop, inplace=True)

    df.rename(columns={'Product/Model #': 'Model'}, inplace=True)

    # Classify SIM information
    df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
    df.drop(columns=['sim_info'], inplace=True)

    if 'Number of Sim' in df.columns:
        if df['Number of Sim'] == 0:
            df["sim_info_status"] = 'uninserted'
        else:
            df["sim_info_status"] = 'inserted'

    # Convert date columns
    for col in ['last_boot_date', 'interval_date', 'active_date']:
        df[col] = df[col].astype(str).apply(convert_arabic_numbers)
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Set warranty status to 'Yes' or 'No' for devices within 30 days from the collection date (Feb 13, 2025)
    warranty_cutoff = pd.to_datetime('2025-02-13') - timedelta(days=30)

    # Assign 'Yes' or 'No' based on warranty conditions, only if the row is NaN
    df['Warranty'] = df.apply(
        lambda row: 'Yes' if pd.isna(row['Warranty']) and (
            row['last_boot_date'] >= warranty_cutoff or 
            row['interval_date'] >= warranty_cutoff or 
            row['active_date'] >= warranty_cutoff) else 'No',
        axis=1
    )
    
    # Compute time differences for churn calculation
    df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.days
    df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.days
    df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.days

    # Drop the original datetime columns after extracting features
    df.drop(columns=['last_boot_date', 'interval_date', 'active_date'], inplace=True)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df

# Function to handle missing values
def handle_missing_values(df):
    # Separate rows with missing values for later prediction
    unknown_data = df[df.isnull().any(axis=1)]
    # Drop rows with missing values for model training
    df_cleaned = df.dropna()
    return df_cleaned, unknown_data

# Function to split data into features and target
def split_features_target(df_cleaned):
    X_cleaned = df_cleaned.drop(columns=['Churn'])
    y_cleaned = df_cleaned['Churn']
    return X_cleaned, y_cleaned

# Objective function for Optuna
def objective(trial, X_train, y_train):
    # Hyperparameter search space for XGBoost
    param_grid = {
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1)
    }

    # Initialize XGBoost with suggested parameters
    xgb_model = xgb.XGBClassifier(random_state=42, **param_grid)
    
    # Perform cross-validation and compute F1 score
    f1_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='f1')
    
    # Return the mean F1 score for the trial
    return f1_scores.mean()

# Function to perform Optuna hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    # Create an Optuna study to maximize the objective function (F1 score)
    study = optuna.create_study(direction='maximize')
    
    # Optimize the hyperparameters
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)
    
    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    
    return best_params

# Function to evaluate model performance
def evaluate_model(model, X_val, y_val, X_test, y_test):
    # Validate on the validation set
    y_val_pred = model.predict(X_val)
    print("Validation Set Classification Report:")
    print(classification_report(y_val, y_val_pred))

    # Test the model
    y_pred = model.predict(X_test)
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned (0)', 'Churned (1)'], 
    #             yticklabels=['Not Churned (0)', 'Churned (1)'])
    # plt.title('Confusion Matrix for Test Set')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.show()

# Function to extract and display feature importance
def display_feature_importances(model, X_cleaned):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 10 most important features:")
    for i in range(10):
        print(f"{X_cleaned.columns[indices[i]]}: {importances[indices[i]]}")

    # # Plot feature importances
    # plt.figure(figsize=(14, 4))
    # plt.title('Feature Importances')
    # plt.barh(range(10), importances[indices[:10]], align="center")
    # plt.yticks(range(10), [X_cleaned.columns[i] for i in indices[:10]])
    # plt.xlabel('Relative Importance')
    # plt.show()

# Main function to run the entire workflow
def main():
    # Load the dataset
    df = load_data("UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
    print("Dataset: ", df.head())
    print("Churn:\n", df["Churn"].head())

    # Preprocess the data
    df_preprocessed = preprocess_data(df)

    # Define churn (0 if interval - activate < 30 days, else don't touch it)
    df_preprocessed['Churn'] = np.where(df_preprocessed['Churn'].isna() & (df_preprocessed['interval - activate'] < 30), 0, df_preprocessed['Churn'])

    # Handle missing values
    df_cleaned, unknown_data = handle_missing_values(df_preprocessed)

    # Split data into features and target
    X_cleaned, y_cleaned = split_features_target(df_cleaned)

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

    # Further split training set (80% train, 20% validation)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_split, y_train_split)

    # Tune hyperparameters using Optuna
    best_params = tune_hyperparameters(X_train_res, y_train_res)

    # Train the XGBoost model with the best hyperparameters found by Optuna
    model = xgb.XGBClassifier(random_state=42, **best_params)
    model.fit(X_train_res, y_train_res)

    # Display top 5 most important features
    display_feature_importances(model, X_cleaned)

    # Evaluate the model
    evaluate_model(model, X_val_split, y_val_split, X_test, y_test)

    # Ensure the prediction set has the same features as the training set
    X_cleaned = X_cleaned[X_train_res.columns]  # Match the columns of the training set

    # Predict missing data churn values
    X_unknown = unknown_data.drop(columns=['Churn'], errors='ignore')
    X_unknown = X_unknown[X_train_res.columns]  # Align columns with training set
    y_unknown = model.predict(X_unknown)

    # Assign predicted churn to unknown data
    unknown_data.loc[:, 'Churn_Predicted'] = y_unknown
    print("Rows with missing data and predicted churn values:")
    print(unknown_data[['Churn', 'Churn_Predicted']].head())

    # Save a copy of the original data (before preprocessing)
    original_df = df.copy()

    # After training the model and predicting churn
    predictions = model.predict(X_cleaned)  # Churn predictions (0 or 1)

    # Ensure you're assigning the predicted churn values back only to the rows used for prediction
    original_df.loc[X_cleaned.index, 'Churn_Predicted'] = predictions

    # Display only the 'Churn' and 'Churn_Predicted' columns for rows in the cleaned data
    print(original_df[['Churn', 'Churn_Predicted']].head())

    original_df.to_csv('./backend/model_building/predictions.csv', index=False)

    # Save the trained model
    joblib.dump(model, './backend/model_building/xgboost_model.joblib')

if __name__ == "__main__":
    main()
