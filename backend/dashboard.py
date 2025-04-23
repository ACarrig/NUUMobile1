import os
import re
import pandas as pd
import json
from fuzzywuzzy import process
from flask import Flask, jsonify, request
from ollama import generate

USERFILES_FOLDER = './backend/userfiles'

column_name_mapping =  {
    'Model': 'Model', 
    'Product/Model #': 'Model', 
    'Product Model': 'Model'
    }

def normalize(counts):
    # Get the list of unique feedback categories
    feature_list = list(counts.keys())
    
    # Create a list of normalized feedback values
    normalized_feature = {}

    # Iterate through the feedback list and match each feedback with the closest one
    for feature in feature_list:

        # Find the best match for the current feedback from the list
        match = process.extractOne(feature, normalized_feature.keys())
        
        if match:  # If a match is found
            best_match, score = match
            # If score is above a threshold, consider it as the same category
            if score >= 80:  # Adjust the threshold as needed
                normalized_feature[best_match] += counts[feature]
                continue  # Skip to next feedback since it has been grouped
        # If no match is found or below threshold, keep the current feedback
        normalized_feature[feature] = normalized_feature.get(feature, 0) + counts[feature]
    
    return normalized_feature

# Function to get sheet names for a specific file
def get_sheet_names(file_name):
    file_path = os.path.join(USERFILES_FOLDER, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_name} was not found in the directory.")

    try:
        # Load the Excel file into a DataFrame
        xls = pd.ExcelFile(file_path)
        # Return sheet names directly as a list, which is serializable
        return xls.sheet_names
    except Exception as e:
        raise Exception(f"Error reading the Excel file: {str(e)}")

# Function to get all columns from a specific sheet in a file
def get_all_columns(file, sheet):
    file_path = os.path.join(USERFILES_FOLDER, file)  # Create the full path to the file

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

# Function to parse carrier names
def extract_json(column):
    try:
        # Check if the entry is a JSON string (sometimes it's just a plain string like 'uninserted')
        if isinstance(column, str) and column.startswith('[{'):
            # If it's a JSON string, load it into a Python list
            data = json.loads(column)
            # Extract the carrier_name from the first item in the list (assuming it's the first slot)
            return data[0].get('name', 'Unknown')
        else:
            return column  # return the raw string for cases like 'uninserted'
    except json.JSONDecodeError:
        return 'Invalid JSON'

def get_column_data(file, sheet, column):   
    file_path = os.path.join(USERFILES_FOLDER, file)  # Create the full path to the file

    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)

    df.columns = get_all_columns(file, sheet)

    if column not in df.columns:
        return {"error": f"Column '{column}' not found in the sheet."}

    frequency_series = df[column].apply(extract_json).value_counts()

    # Convert keys to strings (e.g., NaN => "NaN") for JSON safety
    frequency = {
        str(k) if pd.notna(k) else "NaN": int(v)
        for k, v in frequency_series.items()
    }

    print("Frequency: ", frequency)

    return {"frequency": frequency}

def get_age_range(file, sheet):
    file_path = os.path.join(USERFILES_FOLDER, file)  # Create the full path to the file
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Check if 'Age Range' column exists before processing
    if "Age Range" in df.columns:
        # Get the frequency of each age range
        age_range_frequency = df["Age Range"].value_counts().to_dict()  # Convert to dictionary for easy display
        return {"age_range_frequency": age_range_frequency}  # Return frequency dictionary
    else:
        return {"age_range_frequency": {}}  # Return an empty dictionary if column is missing

def get_model_type(file, sheet):
    file_path = os.path.join(USERFILES_FOLDER, file)  # Create the full path to the file

    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = get_all_columns(file, sheet)
    
    # Check if 'Model' column exists before processing
    if "Model" in df.columns:
        # Get the frequency of each model type
        model_type = normalize(df["Model"].value_counts().to_dict())

        return {"model": model_type}  # Return frequency dictionary
    else:
        return {"model": {}}  # Return an empty dictionary if column is missing
    
def get_model_performance_by_channel(file, sheet):
    file_path = os.path.join(USERFILES_FOLDER, file)
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = get_all_columns(file, sheet)
    
    # Check if 'Model' and either 'Sale Channel' or 'Source' columns exist before processing
    if "Model" in df.columns and ("Sale Channel" in df.columns or "Source" in df.columns):
        # Normalize model names: remove spaces and use title case
        df["Model"] = (
            df["Model"]
            .str.strip()
            .str.replace(" ", "", regex=True)
            .str.lower()
            .replace({"budsa": "earbudsa", "budsb": "earbudsb"})
            .str.title()
        )

        # Use 'Sale Channel' if it exists, otherwise use 'Source'
        channel_column = 'Sale Channel' if 'Sale Channel' in df.columns else 'Source'

        # Group by 'Model' and the selected channel column, and get the count
        model_channel_performance = df.groupby(['Model', channel_column]).size().reset_index(name='Count')

        # Convert to dictionary format suitable for the frontend
        performance_dict = {}
        for _, row in model_channel_performance.iterrows():
            if row['Model'] not in performance_dict:
                performance_dict[row['Model']] = {}
            performance_dict[row['Model']][row[channel_column]] = row['Count']
            
        return {"model_channel_performance": performance_dict}  # Return performance data
    
    else:
        return {"model_channel_performance": {}}  # Return empty if necessary columns are missing
    
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"

# Helper function to get a summary from the AI model about data
def ai_summary(file, sheet, column):
    # Load the Excel file and sheet
    file_path = os.path.join(USERFILES_FOLDER, file)
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Ensure columns are correctly named
    df.columns = get_all_columns(file, sheet)

    # Get the unique values and their counts from the specified column
    unique_data = df[column].value_counts().to_dict()

    # Prepare the prompt for the AI model
    prompt = f"""
    You are analyzing the '{column}' column in returned device data.

    Data: {str(unique_data)}

    Focus on the most frequent values. Do not list counts. Instead:
    - Identify the dominant themes and unusual entries.
    - Interpret what these patterns shows about the customer behavior
    - Relate trends to customer behavior or product quality.

    Be sharp, insightful, and under 60 words. Avoid repeating the data—explain what it means and why it matters. 
    Avoid exaggeration or unsupported speculation. Keep it balanced and neutral.
    """
    
    # Call the AI model to get the summary
    model_response = generate(MODEL_NAME, prompt)  # Assuming generate handles the API call to the model
    
    # Extract the summary from the model's response
    ai_sum = model_response.get('response', 'No summary available')

    print("Summary: ", ai_sum)
    
    # Return the summary as a JSON response
    return jsonify({'summary': ai_sum})

# Helper function to get a summary from the AI model about data
def ai_summary2(file, sheet, column1, column2):
    # Load the Excel file and sheet
    file_path = os.path.join(USERFILES_FOLDER, file)
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Ensure columns are correctly named
    df.columns = get_all_columns(file, sheet)  # Assuming get_all_columns fetches correct column names

    # Get the unique values and their counts from the specified columns
    unique_data1 = df[column1].value_counts().to_dict()
    unique_data2 = df[column2].value_counts().to_dict()

    # Prepare the prompt for the AI model
    prompt = f"""
    You are analyzing the '{column1}' column and '{column2}' column in data.

    Data1: {str(unique_data1)}
    Data2: {str(unique_data2)}

    Focus on the most frequent and unique values. Do not list counts. Instead:
    - Identify the dominant themes and unusual entries.
    - Interpret what these patterns shows about the customer behavior
    - Suggest potential causes, quality/process issues, or red flags worth investigating.
    - If possible, link trends to customer behavior or product quality.

    Be sharp, insightful, and under 100 words. Avoid repeating the data—explain what it means and why it matters.
    """

    # Call the AI model to get the summary
    model_response = generate(MODEL_NAME, prompt)  # Assuming generate handles the API call to the model
    
    # Extract the summary from the model's response
    ai_sum = model_response.get('response', 'No summary available')

    # print("Summary: ", ai_sum)

    # Return the summary as a JSON response
    return jsonify({'summary': ai_sum})

# def comparison_summary(file, sheet, columns):
#     # Load the Excel file and sheet
#     file_path = os.path.join(directory, file)
#     xls = pd.ExcelFile(file_path)
#     df = pd.read_excel(xls, sheet_name=sheet)
    
#     # Ensure columns are correctly named
#     df.columns = get_all_columns(file, sheet)  # Assuming get_all_columns fetches correct column names

#     # Analyze the unique values and counts for each specified column
#     summary = ""
#     for column in columns:
#         unique_data = df[column].value_counts().to_dict()
#         most_common_value = max(unique_data, key=unique_data.get, default="No data in column")
#         most_common_value_count = unique_data.get(most_common_value, 0)
#         summary += f"Most common value in '{column}': {most_common_value} ({most_common_value_count} times).\n"

#     # Prepare the analysis prompt dynamically based on the column names
#     prompt = f"""
#     Given the following data patterns:
#     {summary}
#     Provide an insightful analysis of these trends:
#     - What insights or patterns could we infer from this? Are there any broader implications for the business or product?
#     - Avoid just analyzing numbers. Focus more on what these trends might tell us about the real-world behavior or operational challenges.
#     - Provide possible **solutions** or **recommendations** to address these challenges. For example, what operational changes could improve **product handling**, **customer satisfaction**, or **return processing**?

#     Keep your response concise and Limit your response to 100 words or less.
#     """

#     # Call the AI model to get the summary
#     model_response = generate(MODEL_NAME, prompt)  # Assuming generate handles the API call to the model
    
#     # Extract the summary from the model's response
#     ai_sum = model_response.get('response', 'No summary available')

#     print("Summary: ", ai_sum)

#     # Return the summary as a JSON response
#     return jsonify({'summary': ai_sum})
