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

    nlist = list(returns_df.columns)
    i = 1 # want to start at one because tuples will add 0 col to excel sheet
    for col in nlist:
        if col == "Defect / Damage type":
            break
        i += 1

    # i is the tuple # of the defect/damage type column

    defect_counts = {}

    # add each defect to our dictionary and have a count for it
    for row in returns_df.itertuples():
        if row[i] not in defect_counts:
            defect_counts[row[i]] = 1
        else:
            defect_counts[row[i]] += 1

    # get defect counts in decreasing order by defect
    sorted_defect_counts = {k: v for k, v in sorted(defect_counts.items(), key=lambda item: item[1], reverse=True)}

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