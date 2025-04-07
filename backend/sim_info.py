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

# Function to clean carrier info by removing everything in parentheses
def clean_carrier_label(label):
    # Use regular expression to remove content within parentheses and the parentheses themselves
    return re.sub(r'\s\([^)]+\)', '', label)

def get_carrier_name_from_1slot(file, sheet, slot):
    directory = './backend/userfiles/'  # Path to user files folder
    file_path = os.path.join(directory, file)  # Create the full path to the file

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file} was not found in the directory.")

    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = dashboard.get_all_columns(file, sheet)
        
        # Initialize empty dictionaries to hold the carrier data
        slot_1_carriers = {}
        slot_2_carriers = {}

        # Process Slot 1 if it exists
        if slot == "Slot 1" and "Slot 1" in df.columns:
            slot_1_carriers = df['Slot 1'].apply(clean_carrier_label).value_counts().to_dict()
            return {"carrier": slot_1_carriers}

        # Process Slot 2 if it exists
        if slot == "Slot 2" and "Slot 2" in df.columns:
            slot_2_carriers = df['Slot 2'].apply(clean_carrier_label).value_counts().to_dict()
            return {"carrier": slot_2_carriers}

    except Exception as e:
        raise Exception(f"Error reading the Excel file: {str(e)}")

def get_carrier_name_from_slot(file, sheet):
    directory = './backend/userfiles/'  # Path to user files folder
    file_path = os.path.join(directory, file)  # Create the full path to the file

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file} was not found in the directory.")

    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = dashboard.get_all_columns(file, sheet)
        
        # Initialize empty dictionaries to hold the carrier data
        slot_1_carriers = {}
        slot_2_carriers = {}

        # Process Slot 1 if it exists
        if "Slot 1" in df.columns:
            slot_1_carriers = df['Slot 1'].apply(clean_carrier_label).value_counts().to_dict()

        # Process Slot 2 if it exists
        if "Slot 2" in df.columns:
            slot_2_carriers = df['Slot 2'].apply(clean_carrier_label).value_counts().to_dict()

        # Combine results from both slots, if both exist
        if slot_1_carriers and slot_2_carriers:
            # Merge the dictionaries and sum the counts for common carriers
            combined_carriers = slot_1_carriers.copy()
            for carrier, count in slot_2_carriers.items():
                combined_carriers[carrier] = combined_carriers.get(carrier, 0) + count
            return {"carrier": combined_carriers}

    except Exception as e:
        raise Exception(f"Error reading the Excel file: {str(e)}")
    
def get_carrier_country(file, sheet):
    directory = './backend/userfiles/'  # Path to user files folder
    file_path = os.path.join(directory, file)  # Create the full path to the file

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file} was not found in the directory.")

    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = dashboard.get_all_columns(file, sheet)
        
        if 'Sim Country' in df.columns:
            # Apply the cleaning function to the feature's values
            country = df['Sim Country'].apply(clean_carrier_label).value_counts().to_dict()

            return {"country": country}

    except Exception as e:
        raise Exception(f"Error reading the Excel file: {str(e)}")