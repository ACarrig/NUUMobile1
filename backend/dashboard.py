import os
import pandas as pd
from flask import Flask, jsonify, request

def get_sheet_names(file_name=None):
    directory = './backend/userfiles/'  # Path to user files folder
    
    # If a file name is provided, fetch sheets for that file
    if file_name:  
        file_path = os.path.join(directory, file_name)
        
        if not os.path.exists(file_path):  # Check if the file exists
            return jsonify({"error": f"File {file_name} not found."}), 404
        
        if file_name.endswith(".xls") or file_name.endswith(".xlsx"):  # Check if it's an Excel file
            try:
                # Load the Excel file into a DataFrame
                xls_file = pd.ExcelFile(file_path)
                sheet_names = xls_file.sheet_names  # Get sheet names
                return jsonify({"sheets": sheet_names})  # Return as JSON
                
            except Exception as e:
                return jsonify({"error": f"Error reading {file_name}: {str(e)}"}), 500
        
        else:
            return jsonify({"error": f"Error: Unsupported file format for {file_name}."}), 400
    
    else:  # If no file name is provided, fetch sheets for all files
        sheet_names_list = []  # List to store sheet names for all files
        for file in os.listdir(directory):
            if file.endswith(".xls") or file.endswith(".xlsx"):  # Process only Excel files
                file_path = os.path.join(directory, file)
                try:
                    xls_file = pd.ExcelFile(file_path)
                    sheet_names = xls_file.sheet_names  # Get sheet names for each file
                    sheet_names_list.append({"file": file, "sheets": sheet_names})  # Add file's sheet names to the list
                except Exception as e:
                    sheet_names_list.append({"file": file, "error": f"Error: {str(e)}"})  # Handle errors

        return jsonify({"files": sheet_names_list})  # Return as JSON for all files