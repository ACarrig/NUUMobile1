import os
import re
import pandas as pd
from flask import Flask, jsonify, request
from ollama import generate

column_name_mapping =  {
    'Model': 'Model', 
    'Product/Model #': 'Model', 
    'Product Model': 'Model'
    }

# Function to get sheet names for a specific file
def get_sheet_names(file_name):
    directory = './backend/userfiles/'  # Path to user files folder
    file_path = os.path.join(directory, file_name)  # Create the full path to the file

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_name} was not found in the directory.")

    try:
        # Load the Excel file into a DataFrame
        xls = pd.ExcelFile(file_path)
        # Return sheet names directly as a list, which is serializable
        return xls.sheet_names
    except Exception as e:
        raise Exception(f"Error reading the Excel file: {str(e)}")

# Function to get sheet names from all files
def get_all_sheet_names():
    directory = './backend/userfiles/'  # Path to user files folder
    all_sheets = []

    for file in os.listdir(directory):
        if file.endswith(".xls") or file.endswith(".xlsx"):  # Check for Excel files
            file_path = os.path.join(directory, file)
            try:
                xls = pd.ExcelFile(file_path)
                all_sheets.append({file: xls.sheet_names})  # Directly append sheet names
            except Exception as e:
                all_sheets.append({file: f"Error reading file: {str(e)}"})

    return all_sheets

# Function to get all columns from a specific sheet in a file
def get_all_columns(file, sheet):
    directory = './backend/userfiles/'  # Path to user files folder
    file_path = os.path.join(directory, file)  # Create the full path to the file

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file} was not found in the directory.")

    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=sheet)

        # print("Before columns: ", df.columns.tolist())
        # Standardize columns based on the column_name_mapping
        corrected_columns = [
            column_name_mapping.get(col, col) for col in df.columns
        ]
        
        # Optionally, use regex to handle more dynamic column name standardization
        corrected_columns = [
            re.sub(r'\s*\(.*\)\s*', '', col)  # This will remove any parentheses and their content
            for col in corrected_columns
        ]
        
        # print("Corrected Column: ", corrected_columns)
        return corrected_columns  # Return the updated column list
    except Exception as e:
        raise Exception(f"Error reading the Excel file: {str(e)}")
    
def get_age_range(file, sheet):
    directory = './backend/userfiles/'  # Path to user files folder
    file_path = os.path.join(directory, file)  # Create the full path to the file

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file} was not found in the directory.")

    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=sheet)
        
        # Check if 'Age Range' column exists before processing
        if "Age Range" in df.columns:
            # Get the frequency of each age range
            age_range_frequency = df["Age Range"].value_counts().to_dict()  # Convert to dictionary for easy display
            return {"age_range_frequency": age_range_frequency}  # Return frequency dictionary
        else:
            return {"age_range_frequency": {}}  # Return an empty dictionary if column is missing
    except Exception as e:
        raise Exception(f"Error reading the Excel file: {str(e)}")

def get_model_type(file, sheet):
    directory = './backend/userfiles/'  # Path to user files folder
    file_path = os.path.join(directory, file)  # Create the full path to the file

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file} was not found in the directory.")

    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = get_all_columns(file, sheet)
        
        # Check if 'Model' column exists before processing
        if "Model" in df.columns:
            # Normalize model names: remove spaces and use title case
            df["Model"] = df["Model"].str.strip().str.replace(" ", "", regex=True).str.lower()
            df["Model"] = df["Model"].replace({"budsa": "earbudsa", "budsb": "earbudsb"}).str.title()

            # Get the frequency of each model type
            model_type = df["Model"].value_counts().to_dict()
            return {"model": model_type}  # Return frequency dictionary
        else:
            return {"model": {}}  # Return an empty dictionary if column is missing
    except Exception as e:
        raise Exception(f"Error reading the Excel file: {str(e)}")

def get_model_performance_by_channel(file, sheet):
    directory = './backend/userfiles/'  # Path to user files folder
    file_path = os.path.join(directory, file)  # Create the full path to the file

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file} was not found in the directory.")

    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = get_all_columns(file, sheet)
        
        # Check if 'Model' and 'Sale Channel' columns exist before processing
        if "Model" in df.columns and "Sale Channel" in df.columns:
            # Normalize model names: remove spaces and use title case
            df["Model"] = df["Model"].str.strip().str.replace(" ", "", regex=True).str.lower()
            df["Model"] = df["Model"].replace({"budsa": "earbudsa", "budsb": "earbudsb"}).str.title()

            # Group by 'Model' and 'Sale Channel' and get the count
            model_channel_performance = df.groupby(['Model', 'Sale Channel']).size().reset_index(name='Count')

            # Convert to dictionary format suitable for the frontend
            performance_dict = {}
            for _, row in model_channel_performance.iterrows():
                if row['Model'] not in performance_dict:
                    performance_dict[row['Model']] = {}
                performance_dict[row['Model']][row['Sale Channel']] = row['Count']
                
            return {"model_channel_performance": performance_dict}  # Return performance data
        
        else:
            return {"model_channel_performance": {}}  # Return empty if necessary columns are missing
    except Exception as e:
        raise Exception(f"Error reading the Excel file: {str(e)}")

# Helper function to get a summary from the AI model about data
def ai_summary(file, sheet, column):
    # Load the Excel file and sheet
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Ensure columns are correctly named
    df.columns = get_all_columns(file, sheet)  # Assuming get_all_columns fetches correct column names
    
    # Check if the requested column exists in the data
    if column not in df.columns:
        return jsonify({'error': f"Column '{column}' not found in the sheet."}), 400
    
    # Get the unique values and their counts from the specified column
    unique_data = df[column].value_counts().to_dict()

    # Prepare the prompt for the AI model
    prompt = f"Provide a summary of the following data from the '{column}' column, highlighting key trends and observations. Focus on the most prevalent groups and any notable patterns in the data. Please keep your response concise and avoid using specific numbers. Data: {unique_data}"
    
    # Call the AI model to get the summary
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    MODEL_NAME = "llama3.2:1b"
    model_response = generate(MODEL_NAME, prompt)  # Assuming generate handles the API call to the model
    
    # Extract the summary from the model's response
    ai_sum = model_response.get('response', 'No summary available')
    
    # Return the summary as a JSON response
    return jsonify({'aiSummary': ai_sum})