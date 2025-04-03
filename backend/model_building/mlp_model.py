import pandas as pd
import numpy as np
import json
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Function to load dataset
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Function to preprocess SIM information
def classify_sim_info(sim_info):
    if isinstance(sim_info, str) and sim_info not in ['Unknown', '']:
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

# Function to remove columns with too many NaN values for Churn = 0
def remove_columns(df, threshold):
    churn_na_mask = df['Churn'].isna()
    churn1_mask = df['Churn'] == 1

    na_null_pct = df[churn_na_mask].isnull().mean()
    churn1_null_pct = df[churn1_mask].isnull().mean()
    null_diff = churn1_null_pct - na_null_pct

    cols_to_drop = null_diff[null_diff < -threshold].index.tolist()
    cols_to_drop = [col for col in cols_to_drop if col != 'Churn']
    
    if cols_to_drop:
        print(f"Columns to be dropped: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
    
    return df

# Function to preprocess data
def preprocess_data(df):
    df.drop(columns='Device number', inplace=True)
    df = remove_columns(df, threshold=0.2)
    
    df['sim_info_status'] = df['sim_info'].apply(classify_sim_info)
    df.drop(columns=['sim_info'], inplace=True)

    for col in ['last_boot_date', 'interval_date', 'active_date']:
        df[col] = df[col].astype(str).apply(convert_arabic_numbers)
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['last_boot - activate'] = (df['last_boot_date'] - df['active_date']).dt.days
    df['interval - last_boot'] = (df['interval_date'] - df['last_boot_date']).dt.days
    df['interval - activate'] = (df['interval_date'] - df['active_date']).dt.days

    df.drop(columns=['last_boot_date', 'interval_date', 'active_date'], inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    return df

# Function to handle missing values
def handle_missing_values(df):
    unknown_data = df[df.isnull().any(axis=1)]
    df_cleaned = df.dropna()
    return df_cleaned, unknown_data

# Function to split data into features and target
def split_features_target(df_cleaned):
    X_cleaned = df_cleaned.drop(columns=['Churn'])
    y_cleaned = df_cleaned['Churn']
    return X_cleaned, y_cleaned

# Function to build the neural network model
def build_nn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to train the neural network model
def train_nn_model(X_train, y_train, X_val, y_val):
    model = build_nn_model(X_train.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=50, batch_size=32, 
                        callbacks=[early_stopping], verbose=1)
    
    return model, history

# Function to evaluate model performance
def evaluate_nn_model(model, X_test, y_test):
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned (0)', 'Churned (1)'], 
                yticklabels=['Not Churned (0)', 'Churned (1)'])
    plt.title('Confusion Matrix for Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Main function
def main():
    df = load_data("UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
    
    df_preprocessed = preprocess_data(df)
    df_preprocessed['Churn'] = np.where(df_preprocessed['Churn'].isna() & (df_preprocessed['interval - activate'] < 30), 0, df_preprocessed['Churn'])

    df_cleaned, unknown_data = handle_missing_values(df_preprocessed)
    X_cleaned, y_cleaned = split_features_target(df_cleaned)

    # Standardize features
    scaler = StandardScaler()
    X_cleaned_scaled = scaler.fit_transform(X_cleaned)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned_scaled, y_cleaned, test_size=0.2, random_state=42)

    # Apply SMOTE to balance classes
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train the neural network
    nn_model, history = train_nn_model(X_train_res, y_train_res, X_test, y_test)

    # Evaluate performance
    evaluate_nn_model(nn_model, X_test, y_test)

    # Save the trained model
    nn_model.save('./backend/model_building/nn_model.h5')
    joblib.dump(scaler, './backend/model_building/scaler.joblib')

if __name__ == "__main__":
    main()
