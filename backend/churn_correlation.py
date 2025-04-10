import os
import io
import base64
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from ollama import generate

import matplotlib
matplotlib.use('Agg')

pd.set_option('future.no_silent_downcasting', True)

def encode_categorical(df, column_name):
    """Encode a categorical column with unique integers."""
    mapping = {value: idx for idx, value in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].replace(mapping)
    return mapping

def clean_boolean_column(df, column_name):
    """Standardize boolean-like columns to numeric values."""
    df[column_name] = df[column_name].replace({
        'TRUE': 1, 'FALSE': 0, True: 1, False: 0, '': -1, ' ': -1
    })

def preprocess_data(df, categorical_columns=[], boolean_columns=[]):
    # For columns w/ lots of data, tricky to make numeric
    # - add each item from list to dict as a key, with value being incrementing i
    # - then do the replace, using the dictionary

    """Perform standard cleaning and encoding for correlation analysis."""
    # Clean boolean-like columns
    for col in boolean_columns:
        if col in df.columns:
            clean_boolean_column(df, col)

    # Encode categorical columns
    for col in categorical_columns:
        if col in df.columns:
            encode_categorical(df, col)

    # Convert datetime columns to numeric timestamp
    for col in df.select_dtypes(include=['datetime', 'datetimetz']).columns:
        df[col] = df[col].astype('int64')  # or .view('int64')

    # Fill NaN values with -1 (for missing data)
    df.fillna(-1, inplace=True)
    return df

def churn_relation(file, sheet):
    """Calculate correlation with the target column from Excel data."""
    # Columns that could be used in correlation matrix (FOR 'DATA' SHEET ONLY):
    # 'Type', 'Sale Channel', 'Model', 'Warranty', 'Customer Service Requested', 
    # 'Number of Sim', 'Sim Country', 'Screen Usage (s)', 'Bluetooth (# of pairs)', 'Wallpaper',
    # 'Registered Email', 'last boot - activate', 'last boot - interval'
    # Columns need to be numeric

    # Implementation / how I made it numeric
    # Sale Channel - could create an int for each different type of sale channel
    # Model - ^^
    # Warranty - 1 = yes, and 0 = no, -1 for blank
    # Sim Country - could parse for the number, and do 0 for uninserted/blank
    # Wifi/Internet Connection - 1 for true, 0 for false, -1 for blank
    # Registered Email - 1 true, 0 false, -1 unknown/blank

    file_path = os.path.join('./backend/userfiles/', file)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file} was not found.")

    try:
        # Read the Excel file and the specified sheet
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=sheet)

        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        boolean_columns = df.select_dtypes(include=['bool']).columns.tolist()

        if categorical_columns is None:
            categorical_columns = ['Sale Channel', 'Model', 'Sim Country']  # Default
        if boolean_columns is None:
            boolean_columns = ['Warranty', 'Wifi/Internet Connection', 'Registered Email']  # Default

        # Map the target column (e.g., 'Type') for churn analysis (1 = Return/Repair, 0 = Non-Return)
        target_column = "Type"
        if target_column in df.columns:
            df[target_column] = df[target_column].replace({'Return': 1, 'Repair': 1}).fillna(0)

        # Preprocess the data (clean and encode columns)
        df = preprocess_data(df, categorical_columns, boolean_columns)

        # Calculate correlation with the target column
        correlation = df.corr()[target_column].sort_values(ascending=False)
        correlation_dict = correlation.round(4).to_dict()

        print(correlation_dict)
        return jsonify({'corr': correlation_dict}), 200
    except Exception as e:
        raise Exception(f"Error churn correlation: {str(e)}")

def churn_corr_heatmap(file, sheet):
    """Generate and return a correlation heatmap image."""
    file_path = os.path.join('./backend/userfiles/', file)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file} was not found.")

    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=sheet)

        # Create 'Churn' column
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].fillna(0).astype(int)
        elif 'Type' in df.columns:
            df['Churn'] = df['Type'].replace({'Return': 1, 'Repair': 0}).fillna(0).astype(int)
        else:
            raise Exception("Neither 'Churn' nor 'Type' column found for churn correlation.")

        df = preprocess_data(df)

        # Filter numeric columns only (this excludes object/categorical ones)
        numeric_columns = df.select_dtypes(include=['number']).columns

        # Convert categorical columns to numeric using one-hot encoding (only for the relevant columns)
        df_encoded = pd.get_dummies(df[numeric_columns], drop_first=True)

        # Correlation on only the relevant numeric columns (including encoded categories)
        corr = df_encoded.corr()

        # Generate heatmap with a smaller, more manageable figure size
        plt.figure(figsize=(8, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)

        plt.tight_layout() 

        # Save the plot to a BytesIO object to convert it to a base64 string
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

        # Close the plot
        plt.close()

        return {"image": img_base64}

    except Exception as e:
        raise Exception(f"Error generating heatmap: {str(e)}")
    
def churn_corr_summary(file, sheet):
    response, status_code = churn_relation(file, sheet)
    
    returns = response.get_json()
    corr_data = returns.get('corr')
    
    MODEL_NAME = "llama3.2:1b"
    prompt = "Pretend you are a data scientist. " \
    "As a test, briefly summarize this dictionary while avoiding exact numbers and " \
    "noting key features about parameter correlation: " + str(corr_data)

    model_response = generate(MODEL_NAME, prompt)
    ai_sum = model_response['response']

    return jsonify({'aiSummary': ai_sum})