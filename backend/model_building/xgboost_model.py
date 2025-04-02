import pandas as pd
import numpy as np
import json
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier

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
    df.drop(columns=['Device number'], inplace=True, errors='ignore')

    # Classify SIM information
    df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
    df.drop(columns=['sim_info'], inplace=True, errors='ignore')

    # Convert date columns
    for col in ['last_boot_date', 'interval_date', 'active_date']:
        df[col] = df[col].astype(str).apply(convert_arabic_numbers)
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Compute time differences
    df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.days
    df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.days
    df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.days
    
    df['usage rate'] = df['interval - activate']/df['last_boot - activate']
    df['usage rate'] = df['usage rate'].replace([np.inf, -np.inf], np.nan)
    df['usage rate'] = df['usage rate'].fillna(0)

    df.drop(columns=['last_boot_date', 'interval_date', 'active_date'], inplace=True)

    # Drop columns disproportionately null in Churn=NA cases
    churn_na_mask = df['Churn'].isna()
    churn1_mask = df['Churn'] == 1

    na_null_pct = df[churn_na_mask].isnull().mean()
    churn1_null_pct = df[churn1_mask].isnull().mean()
    null_diff = churn1_null_pct - na_null_pct

    cols_to_drop = null_diff[null_diff < -0.5].index.tolist()
    cols_to_drop = [col for col in cols_to_drop if col != 'Churn']

    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    df = pd.get_dummies(df, drop_first=True)
    
    return df

# Function to handle missing values
def handle_missing_values(df):
    unknown_data = df[df.isnull().any(axis=1)].copy()
    df_cleaned = df.dropna().copy()
    return df_cleaned, unknown_data

# Function to split data into features and target
def split_features_target(df_cleaned):
    X_cleaned = df_cleaned.drop(columns=['Churn'])
    y_cleaned = df_cleaned['Churn']
    return X_cleaned, y_cleaned

# Hyperparameter tuning using GridSearchCV for XGBoost
def tune_xgboost(X_train, y_train):
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    # XGBoost hyperparameter grid
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [8, 10, 12],
        'learning_rate': [0.001, 0.01, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.4, 0.5],
        'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1),  sum(y_train == 0) / sum(y_train == 1) * 2]  # Handle imbalance
    }

    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='f1', cv=5)
    grid_search.fit(X_train, y_train)
    print(f"Best XGBoost Params: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Function to train models
def train_xgboost(X_train, y_train, best_params):
    # Using best hyperparameters from grid search
    xgb = XGBClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        scale_pos_weight=best_params['scale_pos_weight'],
        random_state=42
    )
    xgb.fit(X_train, y_train)
    return xgb

def find_best_threshold(model, X_val, y_val):
    y_val_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)

    # Compute F1 score for different thresholds
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]

    print(f"Best threshold based on F1-score: {best_threshold:.2f}")
    return best_threshold

# Function to adjust threshold for classification
def adjust_threshold(model, X_test, threshold):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return (y_pred_proba >= threshold).astype(int)

def evaluate_model(model, X_val, y_val, X_test, y_test, threshold):
    # Use adjusted threshold for validation and test predictions
    y_val_pred = adjust_threshold(model, X_val, threshold)
    print("Validation Set Report:")
    print(classification_report(y_val, y_val_pred))

    y_pred = adjust_threshold(model, X_test, threshold)
    print("Test Set Report:")
    print(classification_report(y_test, y_pred))

    # Compute and print the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (Test Set):")
    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Threshold = {threshold})')
    plt.show()

# Function to save the trained model
def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Model saved successfully at {model_path}")

# Main function
def main():
    df = load_data("../UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")

    df_preprocessed = preprocess_data(df)

    df_preprocessed['Churn'] = np.where(df_preprocessed['Churn'].isna() & (df_preprocessed['interval - activate'] < 30), 0, df_preprocessed['Churn'])

    df_cleaned, unknown_data = handle_missing_values(df_preprocessed)

    X_cleaned, y_cleaned = split_features_target(df_cleaned)

    # Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Use SMOTEENN to oversample minority and clean majority class
    smoteenn = SMOTEENN(random_state=42)
    X_train_res, y_train_res = smoteenn.fit_resample(X_train, y_train)

    # smote = SMOTE(random_state=42)
    # X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Hyperparameter tuning for XGBoost
    print("Tuning XGBoost...")
    best_xgb_model = tune_xgboost(X_train_res, y_train_res)

    # Train the model with the best hyperparameters
    print("Training XGBoost with best parameters...")
    best_params = best_xgb_model.get_params()
    xgb_model = train_xgboost(X_train_res, y_train_res, best_params)

    threshold = find_best_threshold(xgb_model, X_val, y_val)
    print(f"Evaluating with threshold {threshold}:")
    evaluate_model(xgb_model, X_val, y_val, X_test, y_test, threshold)
    
    predictions = xgb_model.predict(X_cleaned)

    original_df = df.copy()
    # Ensure you're assigning the predicted churn values back only to the rows used for prediction
    original_df.loc[X_cleaned.index, 'Churn_Predicted'] = predictions

    # Display only the 'Churn' and 'Churn_Predicted' columns for rows in the cleaned data
    print(original_df[['Churn', 'Churn_Predicted']].head())

    # Save the best model
    save_model(best_xgb_model, "./backend/model_building/xgboost_model.joblib")

# Run the main function
if __name__ == "__main__":
    main()
