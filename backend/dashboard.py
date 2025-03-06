import os
import pandas as pd
from flask import Flask, jsonify, request

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