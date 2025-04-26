import numpy as np
import pandas as pd
import json, re, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import StepLR

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

def encode_categorical(df, categorical_columns, label_encoders=None):
    """
    Encode categorical columns with LabelEncoder for TabNet input.
    If label_encoders dict is provided, use it; otherwise, fit new encoders.
    Returns transformed df and the dict of encoders.
    """
    if label_encoders is None:
        label_encoders = {}

    for col in categorical_columns:
        le = label_encoders.get(col, LabelEncoder())
        # Fill NA before encoding
        df[col] = df[col].fillna('NA')
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders

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

    # Drop any rows missing Churn
    df_clean = df_combined.dropna(subset=['Churn'])

    # Separate features and target
    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn'].astype(int)

    # Identify categorical columns for TabNet (strings or low unique counts)
    categorical_columns = [col for col in X.columns if X[col].dtype == 'object' or X[col].nunique() < 50]
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    # Encode categorical features with LabelEncoder
    X_encoded, label_encoders = encode_categorical(X.copy(), categorical_columns)

    # Fill missing numeric values with median
    for col in numeric_columns:
        median_val = X_encoded[col].median()
        X_encoded[col] = X_encoded[col].fillna(median_val)

    # Train/test split with stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Further split train into train/val
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Apply SMOTE oversampling only on train split
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_split, y_train_split)

    # TabNet expects numpy arrays
    X_res_np = X_res.values
    y_res_np = y_res.values.reshape(-1, )

    X_val_np = X_val.values
    y_val_np = y_val.values.reshape(-1, )

    X_test_np = X_test.values
    y_test_np = y_test.values.reshape(-1, )

    # Categorical feature indices for TabNet (positions in the numpy array)
    cat_idxs = [X_encoded.columns.get_loc(col) for col in categorical_columns]
    cat_dims = [int(X_encoded[col].nunique()) for col in categorical_columns]

    print(f"Categorical feature indices: {cat_idxs}")
    print(f"Categorical feature dimensions: {cat_dims}")

    # Initialize TabNetClassifier
    clf = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=10,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=StepLR,
        mask_type='entmax'  # "sparsemax" or "entmax"
    )

    # Train TabNet
    clf.fit(
        X_res_np, y_res_np,
        eval_set=[(X_val_np, y_val_np)],
        eval_metric=['auc'],
        max_epochs=100,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    # Predict on validation set
    y_val_probs = clf.predict_proba(X_val_np)[:, 1]
    threshold = 0.5
    y_val_pred_thresh = (y_val_probs > threshold).astype(int)

    print(f"Validation Accuracy (threshold={threshold}):", accuracy_score(y_val_np, y_val_pred_thresh))
    print("Confusion Matrix:\n", confusion_matrix(y_val_np, y_val_pred_thresh))
    print("Classification Report:\n", classification_report(y_val_np, y_val_pred_thresh))

    # Predict on test set
    y_test_probs = clf.predict_proba(X_test_np)[:, 1]
    y_test_pred_thresh = (y_test_probs > threshold).astype(int)

    print(f"Test Accuracy (threshold={threshold}):", accuracy_score(y_test_np, y_test_pred_thresh))
    print("Test Confusion Matrix:\n", confusion_matrix(y_test_np, y_test_pred_thresh))
    print("Test Classification Report:\n", classification_report(y_test_np, y_test_pred_thresh))

    # Prediction function to use later on new dataframes
    def make_predictions(df):
        df_prep = preprocess_data(df)
        X_pred = df_prep.drop(columns=['Churn'], errors='ignore')

        # Encode categoricals with stored encoders, fill missing with 'NA'
        for col in categorical_columns:
            if col in X_pred.columns:
                le = label_encoders[col]
                X_pred[col] = X_pred[col].fillna('NA')
                X_pred[col] = le.transform(X_pred[col].astype(str))
            else:
                # If column missing in new data, fill with zero (or median)
                X_pred[col] = 0

        # Fill missing numeric with median from train
        for col in numeric_columns:
            if col in X_pred.columns:
                median_val = X_encoded[col].median()
                X_pred[col] = X_pred[col].fillna(median_val)
            else:
                X_pred[col] = 0

        # Align columns (add missing columns)
        missing_cols = set(X_encoded.columns) - set(X_pred.columns)
        for col in missing_cols:
            X_pred[col] = 0
        X_pred = X_pred[X_encoded.columns]

        X_pred_np = X_pred.values

        probs = clf.predict_proba(X_pred_np)[:, 1]
        preds = (probs > threshold).astype(int)
        return probs, preds

    # Example usage for new data
    df_new = pd.read_excel("UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
    df_new_preprocessed = preprocess_data(df_new)
    df_eval = df_new_preprocessed.dropna(subset=['Churn'])

    probs_new, preds_new = make_predictions(df_eval)

    y_true = df_eval['Churn'].astype(int)

    print("Accuracy:", accuracy_score(y_true, preds_new))
    print("Confusion Matrix:\n", confusion_matrix(y_true, preds_new))
    print("Classification Report:\n", classification_report(y_true, preds_new))

    # Save model + encoders + columns info
    joblib.dump({
        'tabnet_model': clf,
        'label_encoders': label_encoders,
        'feature_names': X_encoded.columns.tolist(),
        'categorical_columns': categorical_columns,
        'numeric_columns': numeric_columns,
    }, './backend/model_building/tabnet_model.joblib')
    print("\nModel saved successfully.")

if __name__ == "__main__":
    main()
