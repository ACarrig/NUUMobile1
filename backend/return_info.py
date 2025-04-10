import os
import re
import pandas as pd
from fuzzywuzzy import process
from flask import Flask, jsonify, request
from ollama import generate

# get number of returns from a sheet
def returns_count(file, sheet):
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, sheet_name=sheet)
    returns_df = df[df['Type'] == 'Return'] # narrow df to only include returns for speed
    
    num_rows = len(returns_df)
    return jsonify({'num_returns': num_rows}), 200

# get info about the reasons why customers returned devices for a particular file
def returns_info(file, sheet):
    xls = pd.ExcelFile(file)
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
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, sheet_name=sheet)
    feedback_df = df[df['Type'] == 'Return']  # narrow df to only include returns for speed

    feedback_df["Feedback"] = feedback_df["Feedback"].replace("Walmart Reurn", "Walmart Return")

    # Get the counts of the feedback
    feedback_counts = feedback_df['Feedback'].value_counts().to_dict()

    # Normalize feedback using fuzzy matching
    normalized_feedback = normalize_feedback(feedback_counts)

    print("Normalized Feedback: ", normalized_feedback)

    return jsonify({'feedback': normalized_feedback}), 200

# generate an ai summary about the device returns for a particular file
def returns_summary(file, sheet):
    response, status_code = returns_info(file, sheet)
    
    returns = response.get_json()
    return_data = returns.get('defects')
    
    MODEL_NAME = "llama3.2:1b"
    prompt = "Pretend you are a data scientist. " \
    "As a test, briefly summarize this dictionary while avoiding exact numbers and " \
    "noting key features about mock device returns data: " + str(return_data)

    model_response = generate(MODEL_NAME, prompt)
    ai_sum = model_response['response']

    return jsonify({'aiSummary': ai_sum})