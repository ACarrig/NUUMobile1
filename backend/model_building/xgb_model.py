import numpy as np
import pandas as pd
import json, re, joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

def convert_arabic_numbers(text):
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"
    return text.translate(str.maketrans(arabic_digits, western_digits)) if isinstance(text, str) else text

def clean_carrier_label(label):
    return re.sub(r'\s\([^)]+\)', '', label)

def preprocess_data(df):
    rename_dict = {
        'Product/Model #': 'Model',
        'last bootl date': 'last_boot_date',
        'interval date': 'interval_date',
        'activate date': 'active_date',
        'Sale Channel': 'Source',
    }
    df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True)

    columns_to_drop = [
        'Device number', 'imei1', 'Month', 'Office Date', 'Office Time In', 'Final Status', 
        'Defect / Damage type', 'Responsible Party', 'Feedback', 'Slot 1', 'Slot 2', 
        'Verification', 'Spare Parts Used if returned', 'App Usage (s)', 
        'last boot - activate', 'last boot - interval', 'activate'
    ]
    df.drop(columns=[c for c in columns_to_drop if c in df.columns], inplace=True)

    if "Model" in df.columns:
        df["Model"] = (df["Model"].astype(str)
                       .str.strip()
                       .str.replace(" ", "", regex=True)
                       .str.lower()
                       .replace({"budsa": "earbudsa", "budsb": "earbudsb"})
                       .str.title())

    if 'Sim Country' in df.columns:
        df['Sim Country'] = df['Sim Country'].apply(clean_carrier_label)

    if 'sim_info' in df.columns:
        df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
        df.drop(columns=['sim_info'], inplace=True)
    elif 'Sim Card' in df.columns:
        df['sim_info_status'] = df['Sim Card'].apply(classify_sim_info)
        df.drop(columns=['Sim Card'], inplace=True)

    date_cols = ['last_boot_date', 'interval_date', 'active_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(convert_arabic_numbers)
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'last_boot_date' in df.columns and 'active_date' in df.columns:
        df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.days
    if 'interval_date' in df.columns and 'last_boot_date' in df.columns:
        df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.days
    if 'interval_date' in df.columns and 'active_date' in df.columns:
        df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.days

    df.drop(columns=[col for col in date_cols if col in df.columns], inplace=True)

    if 'Type' in df.columns:
        df['Churn'] = np.where(df['Type'] == 'Return', 1, np.where(df['Type'] == 'Repair', 0, np.nan))
        df.drop(columns=['Type'], inplace=True)

    return df

def main():
    # Load data
    df1 = pd.read_excel("UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
    df2 = pd.read_excel("UW_Churn_Pred_Data.xls", sheet_name="Data")

    df1_preprocessed = preprocess_data(df1)
    df2_preprocessed = preprocess_data(df2)

    # Combine datasets
    df_combined = pd.concat([df1_preprocessed, df2_preprocessed], ignore_index=True, sort=False)

    # Fill missing Churn with logic (interval - activate < 30 => churn)
    df_combined['Churn'] = df_combined['Churn'].where(
        df_combined['Churn'].notna(),
        (df_combined['interval - activate'] < 30).astype(int)
    )

    # Drop any rows missing Churn or features
    df_clean = df_combined.dropna(subset=['Churn'])

    # Separate features and target
    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn']

    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Train/test split with stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Further split train into train/val
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Fill missing values (median imputation)
    X_train_split = X_train_split.fillna(X_train_split.median())
    X_val = X_val.fillna(X_train_split.median())

    # Apply SMOTE oversampling only on train split
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_split, y_train_split)

    # Calculate scale_pos_weight for imbalance in original training data
    neg = sum(y_train_split == 0)
    pos = sum(y_train_split == 1)
    scale_pos_weight = neg / pos

    print(f"scale_pos_weight used in XGBClassifier: {scale_pos_weight:.2f}")

    # Define a more focused hyperparameter grid around typical good values
    param_dist = {
        'n_estimators': randint(100, 400),           
        'learning_rate': uniform(0.01, 0.1),         
        'max_depth': randint(3, 7),                   
        'min_child_weight': randint(1, 6),            
        'subsample': uniform(0.7, 0.3),                
        'colsample_bytree': uniform(0.7, 0.3),        
        'gamma': uniform(0, 0.3),                     
    }

    # Base model with scale_pos_weight
    base_model = XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    # Use RandomizedSearchCV with increased iterations and 5-fold CV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=200,              
        scoring='f1',            
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Fit on oversampled training data
    random_search.fit(X_res, y_res)

    print("Best parameters: ", random_search.best_estimator_)

    # Retrieve best model
    model = random_search.best_estimator_

    # Final training on full oversampled training set with evaluation on validation
    model.fit(X_res, y_res, eval_set=[(X_val, y_val)], verbose=True)

    # Predict probabilities on validation set
    y_val_probs = model.predict_proba(X_val)[:, 1]

    # Try threshold tuning to improve class 0 recall
    threshold = 0.4  # You can tune this value between 0.2 and 0.5 based on results
    y_val_pred_thresh = (y_val_probs > threshold).astype(int)

    print(f"Validation Accuracy (threshold={threshold}):", accuracy_score(y_val, y_val_pred_thresh))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred_thresh))
    print("Classification Report:\n", classification_report(y_val, y_val_pred_thresh))

    # Final evaluation on test set using the same threshold
    X_test = X_test.fillna(X_train_split.median())  # Use train median to fill test missing values
    y_test_probs = model.predict_proba(X_test)[:, 1]
    y_test_pred_thresh = (y_test_probs > threshold).astype(int)

    print(f"Test Accuracy (threshold={threshold}):", accuracy_score(y_test, y_test_pred_thresh))
    print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_thresh))
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred_thresh))

    joblib.dump({
        'xgb': model,
        'feature_names': X_encoded.columns.tolist(),
        'median': X_train_split.median()
    }, './backend/model_building/xgb_model.joblib')
    print("\nModel saved successfully.")

    # Prediction function to be used on new dataframes
    def make_predictions(df):
        df_prep = preprocess_data(df)
        df_prep = df_prep.fillna(np.nan)  # Keep NaNs, to be handled in encoding

        X_pred = df_prep.drop(columns=['Churn'], errors='ignore')
        X_pred_encoded = pd.get_dummies(X_pred, drop_first=True)

        # Align features with training set
        missing_cols = set(X_encoded.columns) - set(X_pred_encoded.columns)
        for col in missing_cols:
            X_pred_encoded[col] = 0
        X_pred_encoded = X_pred_encoded[X_encoded.columns]

        # Fill missing values with median from train
        X_pred_encoded = X_pred_encoded.fillna(X_train_split.median())

        probs = model.predict_proba(X_pred_encoded)[:, 1]
        preds = (probs > threshold).astype(int)
        return probs, preds

    # Example usage for new data
    # df_new = pd.read_excel("UW_Churn_Pred_Data.xls", sheet_name="Data")
    df_new = pd.read_excel("UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
    df_new_preprocessed = preprocess_data(df_new)
    df_eval = df_new_preprocessed.dropna(subset=['Churn'])

    probs_new, preds_new = make_predictions(df_eval)

    y_true = df_eval['Churn']

    print("Accuracy:", accuracy_score(y_true, preds_new))
    print("Confusion Matrix:\n", confusion_matrix(y_true, preds_new))
    print("Classification Report:\n", classification_report(y_true, preds_new))

if __name__ == "__main__":
    main()