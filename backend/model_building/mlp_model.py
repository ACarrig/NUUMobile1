import re
import pandas as pd
import numpy as np
import json
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Function to load dataset
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Function to classify SIM information
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

# Convert Arabic numerals
def convert_arabic_numbers(text):
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"
    return text.translate(str.maketrans(arabic_digits, western_digits)) if isinstance(text, str) else text

# Clean carrier info
def clean_carrier_label(label):
    return re.sub(r'\s\([^)]+\)', '', label)

# Preprocessing function
def preprocess_data(df):
    # Rename columns
    rename_dict = {
        'Product/Model #': 'Model',
        'last bootl date': 'last_boot_date',
        'interval date': 'interval_date',
        'activate date': 'active_date',
        'Sale Channel': 'Source',
    }
    df.rename(columns={key: value for key, value in rename_dict.items() if key in df.columns}, inplace=True)

    # Drop columns
    columns_to_drop = [
        'Device number', 'imei1', 'Month', 'Office Date', 'Office Time In', 'Final Status', 
        'Defect / Damage type', 'Responsible Party', 'Feedback', 'Slot 1', 'Slot 2', 
        'Verification', 'Spare Parts Used if returned', 'App Usage (s)', 
        'last boot - activate', 'last boot - interval', 'activate'
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Model name processing
    if "Model" in df.columns:
        df["Model"] = df["Model"].str.strip().str.replace(" ", "", regex=True).str.lower().replace({"budsa": "earbudsa", "budsb": "earbudsb"}).str.title()

    # SIM country processing
    if 'Sim Country' in df.columns:
        df['Sim Country'] = df['Sim Country'].apply(clean_carrier_label)

    # SIM info processing
    if 'sim_info' in df.columns:
        df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
        df.drop(columns=['sim_info'], inplace=True)
    elif 'Sim Card' in df.columns:
        df['sim_info_status'] = df['Sim Card'].apply(classify_sim_info)
        df.drop(columns=['Sim Card'], inplace=True)

    # Date processing
    date_columns = ['last_boot_date', 'interval_date', 'active_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(convert_arabic_numbers)
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Time delta features
    if 'last_boot_date' in df.columns and 'active_date' in df.columns:
        df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.days
    if 'interval_date' in df.columns and 'last_boot_date' in df.columns:
        df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.days
    if 'interval_date' in df.columns and 'active_date' in df.columns:
        df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.days

    # Drop original date columns
    df.drop(columns=[col for col in date_columns if col in df.columns], inplace=True)

    # Churn column creation
    if 'Type' in df.columns:
        df['Churn'] = np.where(df['Type'] == 'Return', 1, np.where(df['Type'] == 'Repair', 0, np.nan))
        df.drop(columns=['Type'], inplace=True)

    if 'Churn' not in df.columns:
        df['Churn'] = np.nan

    # Warranty handling
    if 'Warranty' in df.columns:
        df['Warranty'] = np.where(df['Churn'].isna() & (df['Warranty'] == "Yes"), 1, df['Warranty'])

    # Encoding
    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].apply(str)
            # Special handling for certain columns
            if col == 'Age Range':
                df[col] = df[col].apply(lambda x: int(x[0]) if isinstance(x, str) else x)
            elif col == 'Source':
                df[col] = df[col].apply(lambda x: "B2C 3rd party" if x == "B2C Amazon" else x)
            df[col] = label_encoder.fit_transform(df[col])

    return df

# Create MLP model
def create_mlp_model(input_dim):
    return MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=300,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True
    )

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("MLP Model Classification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("MLP Model Confusion Matrix:\n", cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churned (0)', 'Churned (1)'], 
                yticklabels=['Not Churned (0)', 'Churned (1)'])
    plt.title('Confusion Matrix for MLP Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def main():
    # Load and preprocess data
    df1 = load_data("./backend/model_building/UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
    df2 = load_data("./backend/model_building/UW_Churn_Pred_Data.xls", sheet_name="Data")
    
    df1_preprocessed = preprocess_data(df1)
    df2_preprocessed = preprocess_data(df2)

    common_cols = list(set(df1_preprocessed.columns) & set(df2_preprocessed.columns))
    df1_preprocessed = df1_preprocessed[common_cols]
    df2_preprocessed = df2_preprocessed[common_cols]

    df_combined = pd.concat([df1_preprocessed, df2_preprocessed])
    
    # Churn imputation
    df_combined['Churn'] = np.where(
        df_combined['Churn'].isna() & (df_combined['interval - activate'] < 30), 0, df_combined['Churn']
    )

    df_cleaned = df_combined.dropna()

    y = df_cleaned['Churn']
    X = df_cleaned.drop(columns=['Churn'])
    
    # Train-test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Handle class imbalance
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Create and train MLP model
    print("\nTraining MLP model...")
    mlp_model = create_mlp_model(X_train_res.shape[1])
    mlp_model.fit(X_train_res, y_train_res)

    # Evaluate on validation set
    print("\nValidation Performance:")
    evaluate_model(mlp_model, X_val, y_val)

    # Final evaluation on test set
    print("\nFinal Test Performance:")
    evaluate_model(mlp_model, X_test, y_test)

    # Save model
    joblib.dump({
        'model': mlp_model,
        'feature_names': list(X.columns)
    }, './backend/model_building/mlp_model.joblib')
    print("\nMLP model saved successfully.")

if __name__ == "__main__":
    main()