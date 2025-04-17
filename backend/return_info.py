import os
import re
import pandas as pd
from fuzzywuzzy import process
from flask import Flask, jsonify, request
import dashboard
from ollama import generate

directory = './backend/userfiles/'  # Path to user files folder

# get number of returns from a sheet
def returns_count(file, sheet):
    file_path = os.path.join(directory, file)
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)
    returns_df = df[df['Type'] == 'Return'] # narrow df to only include returns for speed
    
    num_rows = len(returns_df)
    return jsonify({'num_returns': num_rows}), 200

# get info about the reasons why customers returned devices for a particular file
def returns_info(file, sheet):
    file_path = os.path.join(directory, file)
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)
    returns_df = df[df['Type'] == 'Return'] # narrow df to only include returns for speed

    sorted_defect_counts = {str(k): v for k, v in returns_df["Defect / Damage type"].value_counts(dropna=True).items()}

    return jsonify({'defects': sorted_defect_counts}), 200

def normalize_feedback(feedback_counts):
    # Get the list of unique feedback categories
    feedback_list = list(feedback_counts.keys())
    
    # Create a list of normalized feedback values
    normalized_feedback = {}

    # Iterate through the feedback list and match each feedback with the closest one
    for feedback in feedback_list:
        # Skip the feedback 'F'
        if feedback == 'F':
            continue

        # Find the best match for the current feedback from the list
        match = process.extractOne(feedback, normalized_feedback.keys())
        
        if match:  # If a match is found
            best_match, score = match
            # If score is above a threshold, consider it as the same category
            if score >= 80:  # Adjust the threshold as needed
                normalized_feedback[best_match] += feedback_counts[feedback]
                continue  # Skip to next feedback since it has been grouped
        # If no match is found or below threshold, keep the current feedback
        normalized_feedback[feedback] = normalized_feedback.get(feedback, 0) + feedback_counts[feedback]
    
    return normalized_feedback

def feedback_info(file, sheet):
    file_path = os.path.join(directory, file)

    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)

    feedback_df = df[df['Type'] == 'Return']  # narrow df to only include returns for speed

    feedback_df["Feedback"] = feedback_df["Feedback"].replace("Walmart Reurn", "Walmart Return")

    # Get the counts of the feedback
    feedback_counts = feedback_df['Feedback'].value_counts().to_dict()

    # Normalize feedback using fuzzy matching
    normalized_feedback = normalize_feedback(feedback_counts)

    # print("Normalized Feedback: ", normalized_feedback)

    return jsonify({'feedback': normalized_feedback}), 200

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

def verification_info(file, sheet):
    file_path = os.path.join(directory, file)

    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)

    vf_df = df[df['Type'] == 'Return']  # narrow df to only include returns for speed

    # Get the counts of the verification
    vf_counts = vf_df['Verification'].value_counts().to_dict()

    # Normalize verification using fuzzy matching
    normalized_vf = normalize(vf_counts)

    # print("Verification: ", normalized_vf)

    return jsonify({'verification': normalized_vf}), 200

def resparty_info(file, sheet):
    file_path = os.path.join(directory, file)

    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)

    rp_df = df[df['Type'] == 'Return']  # narrow df to only include returns for speed

    # Get the counts of the verification
    rp_counts = rp_df['Responsible Party'].value_counts().to_dict()

    # Normalize verification using fuzzy matching
    normalized_rp = normalize(rp_counts)

    # print("Responsible Party: ", normalized_rp)

    return jsonify({'responsible_party': normalized_rp}), 200

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"

# generate an ai summary about the device returns for a particular file
def returns_summary(file, sheet):

    # Define columns and their meanings
    columns_info = {
        "Feedback": "Customer feedback",
        "Verification": "Verified issues from our testing team",
        "Defect / Damage type": "Damage type",
        "Responsible Party": "The reason why the device is returned"
    }

    # Load the Excel file and sheet
    file_path = os.path.join(directory, file)
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)

    # Ensure columns are correctly named
    df.columns = dashboard.get_all_columns(file, sheet)  # assuming column fix

    summary_details = ""
    for column, meaning in columns_info.items():
        if column in df.columns:
            top_values = df[column].value_counts().nlargest(1).to_dict()
            value_lines = "\n".join([f"  - {val} ({count} times)" for val, count in top_values.items()])
            summary_details += f"\n**{column}** ({meaning}):\n{value_lines}\n"
        else:
            summary_details += f"\n**{column}** ({meaning}): Column not found.\n"

    # Build the AI prompt
    prompt = f"""
    You are analyzing returned device data from a smartphone company. Below is the top occurrence in each relevant column:

    {summary_details}

    Write a professional, insightful, and concise summary that:
    - Clearly interprets **why** these top issues are occurring based on the categories.
    - Connects the data to potential **underlying causes** (e.g., training gaps, design flaws, process inconsistencies).
    - Identifies any **patterns or systemic weaknesses** (e.g., repeated verification failures or poor packaging).
    - Recommends **specific, actionable solutions** (not general statements like "improve communication").

    Your answer should:
    - Be limited to **120 words**.
    - Focus on interpreting **why** these issues exist, not just listing them.
    - Be clear, concise, and grounded in the provided data.
    """

    # Call the AI model to get the summary
    model_response = generate(MODEL_NAME, prompt)  # Assuming generate handles the AI model call
    ai_sum = model_response.get('response', 'No summary available')

    print("Comparison Summary:\n", ai_sum)

    return jsonify({'summary': ai_sum})