import pandas as pd
import numpy as np
import json
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

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

# Define PyTorch dataset
class ChurnDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# Define PyTorch Neural Network Model
class ChurnNN(nn.Module):
    def __init__(self, input_dim):
        super(ChurnNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Function to train the PyTorch model with early stopping and learning rate scheduling
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), './backend/model_building/nn_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Learning rate scheduler step
        scheduler.step()

    return model

# Function to evaluate the model
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    y_pred = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).cpu().numpy()
            y_pred.extend((outputs > 0.5).astype(int))
            y_true.extend(y_batch.numpy())

    print("Test Set Classification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix: \n", cm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned (0)', 'Churned (1)'], 
                yticklabels=['Not Churned (0)', 'Churned (1)'])
    plt.title('Confusion Matrix for Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color='b', label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data("UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
    
    df_preprocessed = preprocess_data(df)
    # Define churn (0 if interval - activate < 30 days, else don't touch it)
    df_preprocessed['Churn'] = np.where(df_preprocessed['Churn'].isna() & (df_preprocessed['interval - activate'] < 30), 0, df_preprocessed['Churn'])
    df_cleaned, _ = handle_missing_values(df_preprocessed)
    X_cleaned, y_cleaned = split_features_target(df_cleaned)

    scaler = StandardScaler()
    X_cleaned_scaled = scaler.fit_transform(X_cleaned)

    # Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned_scaled, y_cleaned, test_size=0.2, random_state=42)

    # Further split train into train (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Apply SMOTE only on the training set
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Create PyTorch datasets
    train_dataset = ChurnDataset(X_train_res, y_train_res)
    val_dataset = ChurnDataset(X_val, y_val)
    test_dataset = ChurnDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ChurnNN(X_train_res.shape[1])
    class_weights = torch.tensor([1.0, 10.0]).to(device)  # Adjust weights according to class distribution
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50)
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
