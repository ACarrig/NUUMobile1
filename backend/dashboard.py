import os
import pandas as pd
from flask import Flask, jsonify, request
from ollama import generate

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
        columns = df.columns.tolist()  # Get all columns as a list
        return columns
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
        
        # Get the frequency of each age range
        age_range_frequency = df["Age Range"].value_counts().to_dict()  # Convert to dictionary for easy display
        print("Age Range Frequency:", age_range_frequency)
        
        return {"age_range_frequency": age_range_frequency}  # Return frequency dictionary
    except Exception as e:
        raise Exception(f"Error reading the Excel file: {str(e)}")
    
# Helper method to get a summary from locally running ai model about data
def age_ai_summary(file, sheet):
    # Call the function and get the response (no need to unpack into response, status_code)
    age_range_data = get_age_range(file, sheet)  # Get only the dictionary from the tuple
    
    # Process the data as needed
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    MODEL_NAME = "llama3.2"

    prompt = prompt = "Provide a summary of the following age range distribution data, highlighting key trends and observations. Focus on the most prevalent age groups and any notable patterns in the data. Please keep your response concise and avoid using specific numbers." + str(age_range_data)
    
    model_response = generate(MODEL_NAME, prompt)
    ai_sum = model_response['response']
    
    return jsonify({'aiSummary': ai_sum})