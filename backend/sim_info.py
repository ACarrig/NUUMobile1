import os
import re
import pandas as pd
import json
import dashboard
from collections import Counter
from flask import Flask, jsonify, request

# Function to parse carrier names
def extract_carrier_name(sim_info):
    try:
        # Check if the entry is a JSON string (sometimes it's just a plain string like 'uninserted')
        if isinstance(sim_info, str) and sim_info.startswith('[{'):
            # If it's a JSON string, load it into a Python list
            data = json.loads(sim_info)
            # Extract the carrier_name from the first item in the list (assuming it's the first slot)
            return data[0].get('carrier_name', 'Unknown')
        else:
            return sim_info  # return the raw string for cases like 'uninserted'
    except json.JSONDecodeError:
        return 'Invalid JSON'

def get_carrier_name(file, sheet):
    directory = './backend/userfiles/'  # Path to user files folder
    file_path = os.path.join(directory, file)  # Create the full path to the file

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file} was not found in the directory.")

    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = dashboard.get_all_columns(file, sheet)

        # Check if 'Model' column exists before processing
        if "sim_info" in df.columns:
            carrier_name = df['sim_info'].apply(extract_carrier_name).value_counts().to_dict()

        return {"carrier": carrier_name}
    except Exception as e:
        raise Exception(f"Error reading the Excel file: {str(e)}")