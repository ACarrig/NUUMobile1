import os
import re
import pandas as pd
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

    sorted_defect_counts = {str(k): v for k, v in returns_df["Defect / Damage type"].value_counts(dropna=False).items()}

    return jsonify({'defects': sorted_defect_counts}), 200

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