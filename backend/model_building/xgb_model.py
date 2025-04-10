import re
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Function to load dataset
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Function to preprocess SIM information
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
    # Rename columns with small differences
    rename_dict = {
        'Product/Model #': 'Model',
        'last bootl date': 'last_boot_date',
        'interval date': 'interval_date',
        'activate date': 'active_date',
        'Sale Channel': 'Source',
    }
    
    df.rename(columns={key: value for key, value in rename_dict.items() if key in df.columns}, inplace=True)

    # Define columns to drop
    columns_to_drop = [
        'Device number', 'Month', 'Office Date', 'Office Time In', 'Final Status', 
        'Defect / Damage type', 'Responsible Party', 'Feedback', 'Slot 1', 'Slot 2', 
        'Verification', 'Spare Parts Used if returned', 'App Usage (s)', 
        'last boot - activate', 'last boot - interval'
    ]
    
    # Drop the columns that exist
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Check if 'Model' column exists before processing
    if "Model" in df.columns:
        # Normalize model names: remove spaces and use title case
        df["Model"] = df["Model"].str.strip().str.replace(" ", "", regex=True).str.lower().replace({"budsa": "earbudsa", "budsb": "earbudsb"}).str.title()

    if 'Sim Country' in df.columns:
        # Apply the cleaning function to the feature's values
        df['Sim Country'] = df['Sim Country'].apply(clean_carrier_label)

    # Classify SIM information if column exists
    if 'sim_info' in df.columns:
        df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
        df.drop(columns=['sim_info'], inplace=True)
    elif 'Sim Card' in df.columns:
        df['sim_info_status'] = df['Sim Card'].apply(classify_sim_info)
        df.drop(columns=['Sim Card'], inplace=True)

    # Convert date columns if they exist
    date_columns = ['last_boot_date', 'interval_date', 'active_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(convert_arabic_numbers)
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Compute time differences for churn calculation
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
        df['Churn'] = np.where(df['Type'] == 'Return', 1, np.where(df['Type'] == 'Repair', 0, np.nan))
        df.drop(columns=['Type'], inplace=True)

    df['Warranty'] = np.where(df['Churn'].isna() & (df['Warranty'] == "Yes"), 1, df['Warranty'])

    # df.to_csv('./backend/model_building/data.csv', index=False)

    # Apply label encoder
    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    for col in categorical_columns:
        if col in df.columns:
            # Handle mixed type columns
            df[col] = df[col].apply(str)  # Convert all values to strings first
            df[col] = label_encoder.fit_transform(df[col])

    return df

# Function to split data into features and target
def split_features_target(df_cleaned):
    X_cleaned = df_cleaned.drop(columns=['Churn'])
    y_cleaned = df_cleaned['Churn']
    return X_cleaned, y_cleaned

# Function to train the XGBoost model
def train_model(X_train, y_train):
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, # number of boosting rounds (trees)
        max_depth=6, # limits how deep each individual tree can go
        learning_rate=0.1, # controls how much each tree influences the final prediction
        subsample=0.8, # fraction of training data used per tree (row sampling)
        colsample_bytree=0.8, # fraction of features used per tree (column sampling)
        eval_metric='logloss', # metric to measure how well the predicted probabilities match the true labels
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='f1')
    print(f"Mean Cross-Validation F1 Score: {cv_scores.mean():.4f}")

    # Train the model
    xgb_model.fit(X_train, y_train)
    return xgb_model

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
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

# Function to train an ensemble model using stacking
def train_ensemble_model(X_train, y_train, model_1, model_2):
    # Define the base models for the stacking
    base_learners = [
        ('xgb_model_1', model_1),
        ('xgb_model_2', model_2)
    ]
    
    # Use logistic regression as the meta-learner => takes the predictions from the base models and learns how to best combine them to make a final prediction
    meta_model = LogisticRegression()
    
    # Create the stacking classifier
    ensemble_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model)
    
    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)
    
    return ensemble_model

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

def get_combined_feature_importance(ensemble_model, model_1, model_2, feature_names_1, feature_names_2):
    """
    Combines feature importances from two models using the ensemble's meta-model weights
    """
    # Get base model importances
    importance_1 = model_1.feature_importances_
    importance_2 = model_2.feature_importances_
    
    # Get meta-model weights (how much each base model contributes)
    meta_weights = ensemble_model.final_estimator_.coef_[0]
    
    # Normalize weights to sum to 1
    norm_weights = meta_weights / np.sum(np.abs(meta_weights))
    
    # Create combined importance dictionary
    combined_importance = {}
    
    # Add Model 1's features (weighted)
    for feature, imp in zip(feature_names_1, importance_1):
        combined_importance[feature] = imp * norm_weights[0]
    
    # Add Model 2's features (weighted)
    for feature, imp in zip(feature_names_2, importance_2):
        if feature in combined_importance:
            combined_importance[feature] += imp * norm_weights[1]
        else:
            combined_importance[feature] = imp * norm_weights[1]
    
    # Convert to DataFrame and sort
    combined_df = pd.DataFrame({
        'Feature': list(combined_importance.keys()),
        'Importance': list(combined_importance.values())
    }).sort_values('Importance', ascending=False)
    
    return combined_df

# Main function to run the ensemble model evaluation
def main():
    # Load the datasets again
    df1 = load_data("UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
    df2 = load_data("UW_Churn_Pred_Data.xls", sheet_name="Data")

    # Preprocess the data
    df1_preprocessed = preprocess_data(df1)
    df2_preprocessed = preprocess_data(df2)

    # Define churn (0 if interval - activate < 30 days, else don't touch it)
    df1_preprocessed['Churn'] = np.where(df1_preprocessed['Churn'].isna() & (df1_preprocessed['interval - activate'] < 30), 0, df1_preprocessed['Churn'])
    df2_preprocessed['Churn'] = np.where(df2_preprocessed['Churn'].isna() & (df2_preprocessed['interval - activate'] < 30), 0, df2_preprocessed['Churn'])

    # Handle missing values
    df1_cleaned = df1_preprocessed.dropna()
    df2_cleaned = df2_preprocessed.dropna()

    # Split data into features and target for df1 and df2
    X_cleaned_1, y_cleaned_1 = split_features_target(df1_cleaned)
    X_cleaned_2, y_cleaned_2 = split_features_target(df2_cleaned)

    # Train-test split for df1 and df2
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_cleaned_1, y_cleaned_1, test_size=0.2, random_state=42)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_cleaned_2, y_cleaned_2, test_size=0.2, random_state=42)

    # Handle class imbalance with SMOTE for df1 and df2
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res_1, y_train_res_1 = smote.fit_resample(X_train_1, y_train_1)
    X_train_res_2, y_train_res_2 = smote.fit_resample(X_train_2, y_train_2)

    # Train the models for df1 and df2
    xgb_model_1 = train_model(X_train_res_1, y_train_res_1)
    xgb_model_2 = train_model(X_train_res_2, y_train_res_2)

    # Train the ensemble model
    ensemble_model = train_ensemble_model(X_train_res_1, y_train_res_1, xgb_model_1, xgb_model_2)
    
    # Evaluate and save the ensemble model
    evaluate_ensemble_model(X_test_1, y_test_1, ensemble_model)

    combined_importance = get_combined_feature_importance(ensemble_model, xgb_model_1, xgb_model_2, X_train_res_1, X_train_res_2)

    print("Feature Importances: ", combined_importance[10:])

    models = {
        'xgb_model_1': xgb_model_1,
        'xgb_model_2': xgb_model_2,
        'ensemble_model': ensemble_model,
    }

    # Save the ensemble model
    joblib.dump(models, './backend/model_building/ensemble_model.joblib')
    print("Ensemble model saved successfully.")

if __name__ == "__main__":
    main()
