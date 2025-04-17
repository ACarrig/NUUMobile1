import re
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Function to load dataset
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Function to classify SIM information as present (1) or not (0)
def classify_sim_info(sim_info):
    if isinstance(sim_info, str) and sim_info not in ['Unknown', '']:
        try:
            parsed = json.loads(sim_info)
            if isinstance(parsed, list) and parsed:
                carrier_name = parsed[0].get('carrier_name', None)
                return 1 if carrier_name and carrier_name != 'Unknown' else 0
        except json.JSONDecodeError:
            return 0
    return 0

# Convert Arabic numerals to Western numerals
def convert_arabic_numbers(text):
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"
    return text.translate(str.maketrans(arabic_digits, western_digits)) if isinstance(text, str) else text

# Function to clean carrier info by removing everything in parentheses
def clean_carrier_label(label):
    # Use regular expression to remove content within parentheses and the parentheses themselves
    return re.sub(r'\s\([^)]+\)', '', label)

# Preprocessing function to clean and prepare the data
def preprocess_data(df):
    # Rename columns to standardized names
    rename_dict = {
        'Product/Model #': 'Model',
        'last bootl date': 'last_boot_date',
        'interval date': 'interval_date',
        'activate date': 'active_date',
        'Sale Channel': 'Source',
    }
    
    # Only rename columns that exist in the dataframe
    df.rename(columns={key: value for key, value in rename_dict.items() if key in df.columns}, inplace=True)

    # Drop irrelevant or unnecessary columns if they exist
    columns_to_drop = [
        'Device number', 'imei1', 'Month', 'Office Date', 'Office Time In', 'Final Status', 
        'Defect / Damage type', 'Responsible Party', 'Feedback', 'Slot 1', 'Slot 2', 
        'Verification', 'Spare Parts Used if returned', 'App Usage (s)', 
        'last boot - activate', 'last boot - interval', 'activate'
    ]
    
    # Drop the columns that exist
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Check if 'Model' column exists before processing
    if "Model" in df.columns:
        # Normalize model names: remove spaces and use title case
        df["Model"] = df["Model"].str.strip().str.replace(" ", "", regex=True).str.lower().replace({"budsa": "earbudsa", "budsb": "earbudsb"}).str.title()

    # Clean the SIM country column values if present
    if 'Sim Country' in df.columns:
        # Apply the cleaning function to the feature's values
        df['Sim Country'] = df['Sim Country'].apply(clean_carrier_label)

    # Classify SIM information if column exists
    if 'sim_info' in df.columns:
        # Classify SIM info and drop original column
        df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
        df.drop(columns=['sim_info'], inplace=True)
    elif 'Sim Card' in df.columns:
        # Handle alternative column name for same data
        df['sim_info_status'] = df['Sim Card'].apply(classify_sim_info)
        df.drop(columns=['Sim Card'], inplace=True)

    # Convert date columns if they exist
    date_columns = ['last_boot_date', 'interval_date', 'active_date']
    for col in date_columns:
        if col in df.columns:
            # Convert any Arabic numerals and then parse as datetime
            df[col] = df[col].astype(str).apply(convert_arabic_numbers)
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Compute time differences (in days) for churn calculation
    if 'last_boot_date' in df.columns and 'active_date' in df.columns:
        df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.days
    if 'interval_date' in df.columns and 'last_boot_date' in df.columns:
        df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.days
    if 'interval_date' in df.columns and 'active_date' in df.columns:
        df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.days

    # Drop the original datetime columns after extracting features
    df.drop(columns=[col for col in date_columns if col in df.columns], inplace=True)

    # Create 'Churn' column based on 'Type'
    if 'Type' in df.columns:
        # Return → 1 (churned), Repair → 0 (not churned), others → NaN
        df['Churn'] = np.where(df['Type'] == 'Return', 1, np.where(df['Type'] == 'Repair', 0, np.nan))
        df.drop(columns=['Type'], inplace=True)

    # If Churn column doesn't exist, add it with all values as NaN
    if 'Churn' not in df.columns:
        df['Churn'] = np.nan

    # Patch missing churn values using warranty information if available
    if 'Warranty' in df.columns:
        df['Warranty'] = np.where(df['Churn'].isna() & (df['Warranty'] == "Yes"), 1, df['Warranty'])

    # df.to_csv('./backend/model_building/data.csv', index=False)

    # Encode all categorical string or boolean columns using LabelEncoder
    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    for col in categorical_columns:
        if col in df.columns:
            # Handle mixed type columns
            df[col] = df[col].apply(str)  # Convert all values to strings first
            df[col] = label_encoder.fit_transform(df[col])

    return df

# Function to define base models
def get_base_models():
    return [
        ('xgb', xgb.XGBClassifier(
            n_estimators=300,  
            max_depth=3,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )),
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )),
        ('logreg', LogisticRegression(
            penalty='l2',
            C=0.1,
            solver='liblinear',
            random_state=42,
            max_iter=1000
        ))
    ]

# Function to train and evaluate base models
def evaluate_base_models(base_models, X_train, y_train, X_val, y_val):
    base_model_performance = {}
    for name, model in base_models:
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        print(f"\n{name} Validation Performance:")
        print(classification_report(y_val, val_pred))
        base_model_performance[name] = classification_report(y_val, val_pred, output_dict=True)
    return base_model_performance

# Function to create and train the stacking ensemble
def create_stacking_ensemble(base_models, X_train, y_train):
    ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(penalty='l2', C=0.1),
        stack_method='predict_proba',
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    ensemble.fit(X_train, y_train)
    return ensemble

# Function to calibrate the ensemble model
def calibrate_ensemble(ensemble, X_val, y_val):
    calibrated_ensemble = CalibratedClassifierCV(
        ensemble,
        method='isotonic',
        cv='prefit'
    )
    calibrated_ensemble.fit(X_val, y_val)
    return calibrated_ensemble

# Function to evaluate and save the ensemble model
def evaluate_ensemble_model(X_test, y_test, ensemble_model):
    # Predict using the ensemble model
    y_pred = ensemble_model.predict(X_test)
    
    # Print the classification report
    print("Ensemble Model Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Ensemble Model Confusion Matrix:\n", cm)
    
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned (0)', 'Churned (1)'], 
    #             yticklabels=['Not Churned (0)', 'Churned (1)'])
    # plt.title('Confusion Matrix for Ensemble Model')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.show()

# Main function to run the ensemble model evaluation
def main():
    # Load and preprocess data
    df1 = load_data("./backend/model_building/UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
    df2 = load_data("./backend/model_building/UW_Churn_Pred_Data.xls", sheet_name="Data")
    
    # Apply the preprocessing pipeline to clean the features
    df1_preprocessed = preprocess_data(df1)
    df2_preprocessed = preprocess_data(df2)

    # Ensure both datasets have the same columns before merging => prevent schema mismatches & build 1 unified dataset for modeling
    common_cols = list(set(df1_preprocessed.columns) & set(df2_preprocessed.columns))
    df1_preprocessed = df1_preprocessed[common_cols]
    df2_preprocessed = df2_preprocessed[common_cols]

    # Combine both preprocessed datasets into 1 dataframe
    df_combined = pd.concat([df1_preprocessed, df2_preprocessed])
    
    # Fill in missing Churn values using the interval-activate heuristic:
    # If interval - activate < 30 days, then consider it as not churned (Churn = 0)
    df_combined['Churn'] = np.where(
        df_combined['Churn'].isna() & (df_combined['interval - activate'] < 30), 0, df_combined['Churn']
    )

    # Remove any rows with missing values after processing
    df_cleaned = df_combined.dropna()

    # Separate target and features
    y = df_cleaned['Churn']
    X = df_cleaned.drop(columns=['Churn'])
    
    # Create a 60/20/20 split for training, validation, and testing
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Handle class imbalance
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Get base models
    base_models = get_base_models()

    # Train and evaluate base models separately using the validation set
    print("Training and evaluating base models...")
    evaluate_base_models(base_models, X_train_res, y_train_res, X_val, y_val)

    # Create a stacking ensemble using the base models and a logistic regression meta-model
    print("\nCreating stacking ensemble...")
    ensemble = create_stacking_ensemble(base_models, X_train_res, y_train_res)

    # Calibrate the ensemble using isotonic regression for better probability estimates
    print("\nCalibrating ensemble...")
    calibrated_ensemble = calibrate_ensemble(ensemble, X_val, y_val)

    # Final evaluation on test set
    print("\nFinal Test Performance:")
    evaluate_ensemble_model(X_test, y_test, calibrated_ensemble)

    # Feature importance analysis
    if hasattr(ensemble.final_estimator_, 'coef_'):
        print("\nMeta-learner Feature Importances (base model contributions):")
        meta_importances = pd.Series(
            np.abs(ensemble.final_estimator_.coef_[0]),
            index=[name for name, _ in base_models]
        )
        print(meta_importances.sort_values(ascending=False))

    # Save all components
    joblib.dump({
        'ensemble': ensemble,
        'base_models': dict(base_models),
        'feature_names': list(X.columns)
    }, './backend/model_building/ensemble_model.joblib')
    print("\nModel saved successfully.")

if __name__ == "__main__":
    main()
