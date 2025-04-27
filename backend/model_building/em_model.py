import numpy as np
import pandas as pd
import json, re, joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Helper functions
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

# Main function
def main():
    # Load and preprocess data (same as before)
    df1 = pd.read_excel("UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
    df2 = pd.read_excel("UW_Churn_Pred_Data.xls", sheet_name="Data")

    df1_preprocessed = preprocess_data(df1)
    df2_preprocessed = preprocess_data(df2)

    df_combined = pd.concat([df1_preprocessed, df2_preprocessed], ignore_index=True, sort=False)
    df_combined['Churn'] = df_combined['Churn'].where(
        df_combined['Churn'].notna(),
        (df_combined['interval - activate'] < 30).astype(int)
    )
    df_clean = df_combined.dropna(subset=['Churn'])

    # Features and target
    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn']

    # One-hot encode
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Further split train/val
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Calculate scale_pos_weight for XGB
    neg = sum(y_train_split == 0)
    pos = sum(y_train_split == 1)
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight used in XGBClassifier: {scale_pos_weight:.2f}")

    # ========== Enhanced Model Definitions with Imputation ==========
    
    # Create an imputer for models that need it
    imputer = SimpleImputer(strategy='median')
    
    # Optimized XGBoost model with calibration
    xgb_model = make_pipeline(
        SimpleImputer(strategy='median'),
        XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_estimators=350,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3
        )
    )
    
    # Calibrate the XGBoost model for better probability estimates
    calibrated_xgb = CalibratedClassifierCV(xgb_model, method='isotonic', cv=3)

    # Optimized Random Forest with class weights
    rf_model = make_pipeline(
        SimpleImputer(strategy='median'),
        RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=4,
            max_features=0.7,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
    )

    # Optimized Logistic Regression with elasticnet penalty
    lr_model = make_pipeline(
        SimpleImputer(strategy='median'),
        LogisticRegression(
            solver='saga',
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
            penalty='elasticnet',
            l1_ratio=0.5,
            C=0.2
        )
    )

    # Additional Gradient Boosting model
    gb_model = make_pipeline(
        SimpleImputer(strategy='median'),
        GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42
        )
    )

    # ========== Ensemble Strategies ==========
    
    # Option 1: Voting Classifier (Soft Voting)
    voting_model = VotingClassifier(
        estimators=[
            ('xgb', calibrated_xgb),
            ('rf', rf_model),
            ('lr', lr_model),
            ('gb', gb_model)
        ],
        voting='soft',
        weights=[3, 2, 1, 2]  # Higher weight for better performing models
    )
    
    # Option 2: Stacking Classifier (More advanced ensemble)
    stack_model = StackingClassifier(
        estimators=[
            ('xgb', calibrated_xgb),
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        final_estimator=make_pipeline(
            SimpleImputer(strategy='median'),
            LogisticRegression(
                class_weight='balanced',
                C=0.1,
                max_iter=2000
            )
        ),
        stack_method='predict_proba',
        n_jobs=-1
    )

    # Create SMOTE pipeline for training
    smote_pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('smote', SMOTE(random_state=42))
    ])

    # Apply SMOTE to training data
    X_res, y_res = smote_pipeline.fit_resample(X_train_split, y_train_split)
    
    # Need to transform the validation set as well (without SMOTE)
    X_val_imputed = smote_pipeline.named_steps['imputer'].transform(X_val)

    # Train both ensemble models
    print("Training Voting Classifier...")
    voting_model.fit(X_res, y_res)
    
    print("Training Stacking Classifier...")
    stack_model.fit(X_res, y_res)

    # Evaluate both ensemble approaches
    for name, model in [('Voting Classifier', voting_model), ('Stacking Classifier', stack_model)]:
        print(f"\n=== {name} Evaluation ===")
        
        # Validation set evaluation
        y_val_probs = model.predict_proba(X_val_imputed)[:, 1]
        for threshold in [0.4, 0.35, 0.45]:  # Test multiple thresholds
            y_val_pred = (y_val_probs > threshold).astype(int)
            print(f"\nValidation (threshold={threshold}):")
            print(f"AUC: {roc_auc_score(y_val, y_val_probs):.4f}")
            print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
            print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
            print("Classification Report:\n", classification_report(y_val, y_val_pred))
        
        # Test set evaluation (using best threshold)
        X_test_imputed = smote_pipeline.named_steps['imputer'].transform(X_test)
        y_test_probs = model.predict_proba(X_test_imputed)[:, 1]
        y_test_pred = (y_test_probs > 0.4).astype(int)
        print("\nTest Set Performance:")
        print(f"AUC: {roc_auc_score(y_test, y_test_probs):.4f}")
        print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
        print("Classification Report:\n", classification_report(y_test, y_test_pred))

    # Select the best performing model (based on your evaluation)
    best_model = stack_model  # Or voting_model if it performed better
    
    # ========== Final Model Optimization ==========
    
    # If you want to further optimize the stacking classifier's meta-model
    print("\nOptimizing the stacking classifier's meta-model...")
    
    # Get the stacked features from the base models
    base_preds = np.column_stack([
        calibrated_xgb.fit(X_res, y_res).predict_proba(X_val_imputed)[:, 1],
        rf_model.fit(X_res, y_res).predict_proba(X_val_imputed)[:, 1],
        gb_model.fit(X_res, y_res).predict_proba(X_val_imputed)[:, 1]
    ])
    
    # Optimize the meta-model (LogisticRegression)
    meta_model = make_pipeline(
        SimpleImputer(strategy='median'),
        LogisticRegression(class_weight='balanced', random_state=42)
    )
    
    param_grid = {
        'logisticregression__C': [0.001, 0.01, 0.1, 1, 10],
        'logisticregression__penalty': ['l1', 'l2', 'elasticnet'],
        'logisticregression__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'logisticregression__solver': ['saga']
    }
    
    search = RandomizedSearchCV(
        meta_model,
        param_distributions=param_grid,
        n_iter=30,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    search.fit(base_preds, y_val)
    
    print("Best meta-model parameters:", search.best_params_)
    
    # Retrain the stacking classifier with optimized meta-model
    optimized_stack = StackingClassifier(
        estimators=[
            ('xgb', calibrated_xgb),
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        final_estimator=search.best_estimator_,
        stack_method='predict_proba',
        n_jobs=-1
    )
    optimized_stack.fit(X_res, y_res)
    
    # Final evaluation
    print("\n=== Optimized Stacking Classifier ===")
    X_test_imputed = smote_pipeline.named_steps['imputer'].transform(X_test)
    y_test_probs = optimized_stack.predict_proba(X_test_imputed)[:, 1]
    y_test_pred = (y_test_probs > 0.4).astype(int)
    print(f"Test AUC: {roc_auc_score(y_test, y_test_probs):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
    
    # ========== Save the Best Model ==========
    joblib.dump({
        'ensemble': optimized_stack,
        'feature_names': X_encoded.columns.tolist(),
        'imputer': smote_pipeline.named_steps['imputer'],
    }, './backend/model_building/ensemble_model.joblib')
    
    print("\nOptimized ensemble model saved successfully.")

    # Prediction function
    def make_predictions(df):
        df_prep = preprocess_data(df)
        df_prep = df_prep.fillna(np.nan)

        X_pred = df_prep.drop(columns=['Churn'], errors='ignore')
        X_pred_encoded = pd.get_dummies(X_pred, drop_first=True)

        missing_cols = set(X_encoded.columns) - set(X_pred_encoded.columns)
        for col in missing_cols:
            X_pred_encoded[col] = 0
        X_pred_encoded = X_pred_encoded[X_encoded.columns]

        # Use the saved imputer to transform new data
        X_pred_imputed = smote_pipeline.named_steps['imputer'].transform(X_pred_encoded)

        probs = optimized_stack.predict_proba(X_pred_imputed)[:, 1]
        preds = (probs > 0.4).astype(int)
        return probs, preds

    # Example new data
    df_new = pd.read_excel("UW_Churn_Pred_Data.xls", sheet_name="Data")
    df_new_preprocessed = preprocess_data(df_new)
    df_eval = df_new_preprocessed.dropna(subset=['Churn'])

    probs_new, preds_new = make_predictions(df_eval)
    y_true = df_eval['Churn']

    print("\nNew Data Evaluation:")
    print("Accuracy:", accuracy_score(y_true, preds_new))
    print("Confusion Matrix:\n", confusion_matrix(y_true, preds_new))
    print("Classification Report:\n", classification_report(y_true, preds_new))

if __name__ == "__main__":
    main()