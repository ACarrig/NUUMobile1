import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from flask_cors import CORS
from ollama import generate

pd.set_option('future.no_silent_downcasting', True)

def churn_relation(file):
    xls = pd.ExcelFile(file)
    sheet = 'Data'
    data_df = pd.read_excel(xls, sheet_name=sheet)

    # Columns that could be used in correlation matrix (FOR 'DATA' SHEET ONLY):
    # 'Type', 'Sale Channel', 'Model', 'Warranty', 'Customer Service Requested', 
    # 'Number of Sim', 'Sim Country', 'Screen Usage (s)', 'Bluetooth (# of pairs)', 'Wallpaper',
    # 'Registered Email', 'last boot - activate', 'last boot - interval'
    # Columns need to be numeric

    # Implementation / how I made it numeric
    # Sale Channel - could create an int for each different type of sale channel
    # Model - ^^
    # Warranty - 1 = yes, and 0 = no, -1 for blank
    # Sim Country - could parse for the number, and do 0 for uninserted/blank
    # Wifi/Internet Connection - 1 for true, 0 for false, -1 for blank
    # Registered Email - 1 true, 0 false, -1 unknown/blank

    saleChannel_to_num = {}
    model_to_num = {}
    sim_to_num = {}

    # Easy to make numeric for TRUE or FALSE columns
    data_df['Type'] = data_df['Type'].replace({'Return': 1, 'Repair': 1}).fillna(0)
    data_df['Warranty'] = data_df['Warranty'].replace({'Yes': 1, 'No': 0, '': -1, ' ': -1})
    data_df['Wifi/Internet Connection'] = data_df['Wifi/Internet Connection'].replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0, '': -1, ' ': -1})
    data_df['Registered Email'] = data_df['Registered Email'].replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0, '': -1, ' ': -1})

    # For columns w/ lots of data, tricky to make numeric
    # - add each item from list to dict as a key, with value being incrementing i
    # - then do the replace, using the dictionary

    # for sale channels
    sale_channels = data_df['Sale Channel'].unique()
    i = 0
    for channel in sale_channels:
        saleChannel_to_num[channel] = i
        i += 1
    data_df['Sale Channel'] = data_df['Sale Channel'].replace(saleChannel_to_num)

    # for phone models
    models = data_df['Model'].unique()
    i = 0
    for model in models:
        model_to_num[model] = i
        i += 1
    data_df['Model'] = data_df['Model'].replace(model_to_num)

    # for country sim cards
    sims = data_df['Sim Country'].unique()
    i = 0
    for sim in sims:
        sim_to_num[sim] = i
        i += 1
    data_df['Sim Country'] = data_df['Sim Country'].replace(sim_to_num)

    data_df = data_df[['Type', 'Sale Channel', 'Model', 'Warranty', 'Customer Service Requested', 'Number of Sim', 'Sim Country', 'Screen Usage (s)', 'Bluetooth (# of pairs)', 'Wifi/Internet Connection','Wallpaper', 'Registered Email', 'last boot - activate', 'last boot - interval']]
    data_df.fillna(-1, inplace=True)

    correlation_matrix = data_df.corr()['Type'].sort_values(ascending=False)
    rounded_list = [round(num, 4) for num in correlation_matrix]
    correlation_dict = dict(zip(correlation_matrix.keys(),rounded_list))

    print(correlation_dict)
    return jsonify({'corr': correlation_dict}), 200

def churn_corr_summary(file):
    response, status_code = churn_relation(file)
    
    returns = response.get_json()
    corr_data = returns.get('corr')
    
    MODEL_NAME = "llama3.2:1b"
    prompt = "Pretend you are a data scientist. " \
    "As a test, briefly summarize this dictionary while avoiding exact numbers and " \
    "noting key features about parameter correlation: " + str(corr_data)

    model_response = generate(MODEL_NAME, prompt)
    ai_sum = model_response['response']

    return jsonify({'aiSummary': ai_sum})