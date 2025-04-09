import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from ollama import generate
import os

def get_df_helper(file, sheet):
    directory = './backend/userfiles/'  # Path to user files folder
    file_path = os.path.join(directory, file)  # Create the full path to the file

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file} was not found in the directory.")

    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=sheet)
        return df
    except Exception as e:
        raise Exception(f"Error reading the Excel file: {str(e)}")
    
# do processing each of these functions would normally need to do to add new columns to df
def preprocess_df(df):
    df.columns = df.columns.str.strip()  # Remove any leading or trailing spaces

    activate_dates = df['activate date'].tolist()
    interval_dates = df['interval date'].tolist()

    # Get interval_dates - activate_dates for time retaining device
    #    can then see how long (on avg.) a device is kept
    intv_actv = []
    for activation, interval in zip(activate_dates, interval_dates):
        if activation != 0 and interval != 0:
            try:
                activate_dt = datetime.strptime(activation, "%Y-%m-%d %H:%M:%S")
                interval_dt = datetime.strptime(interval, "%Y-%m-%d %H:%M:%S")

                intv_actv.append((interval_dt - activate_dt).seconds / 3600)
            except:
                intv_actv.append(0)
        else:
            intv_actv.append(0)

    # Get activation month
    activation_month = []
    month_conv = {
        12: 'December',
        11: 'November',
        10: 'October',
        9: 'September',
        8: 'August',
        7: 'July',
        6: 'June',
        5: 'May',
        4: 'April',
        3: 'March',
        2: 'February',
        1: 'January'
    }

    for activation in activate_dates:
        if activation != 0:
            try:
                activate_dt = datetime.strptime(activation, "%Y-%m-%d %H:%M:%S")
                activation_month.append(month_conv[activate_dt.month])
            except:
                activation_month.append('None')
        else:
            activation_month.append('None')

    # Add intv_active and activation_month to dataframe
    df['activation_month'] = activation_month
    df['interval - activate'] = intv_actv

    return df


# get the number of sales by monthy
def monthly_sales(file, sheet):
    df = get_df_helper(file, sheet)
    df = preprocess_df(df)
    
    # Total sales by month
    month_dict = {
        'July': 0,
        'August': 0,
        'September': 0,
        'October': 0,
        'November': 0
    }

    for month in list(month_dict.keys()):
        temp_df = df[df['activation_month'] == month]
        month_dict[month] = len(temp_df)

    return jsonify({'monthlySales': month_dict}), 200

# get the number of sales of each model, by month
def monthly_model_sales(file, sheet):
    df = get_df_helper(file, sheet)
    df = preprocess_df(df)

    # Sales of models by month
    all_models = list(df['Model'].unique())

    month_model_sales_dict = {
        'January': {},
        'February': {},
        'March': {},
        'April': {},
        'May': {},
        'June': {},
        'July': {},
        'August': {},
        'September': {},
        'October': {},
        'November': {},
        'December': {}
    }

    # create temp_df containing rows with certain activation month of device
    for month in list(month_model_sales_dict.keys()):
        temp_df = df[df['activation_month'] == month]
        month_model_sales_dict[month] = temp_df['Model'].value_counts().to_dict()

    return jsonify({'modelSales': month_model_sales_dict}), 200

# get the average interval - activate time per month of any device
def device_retainment(file, sheet):
    df = get_df_helper(file, sheet)
    df = preprocess_df(df)

    # average device retainment time by month
    month_time_dict = {
        'July': 0,
        'August': 0,
        'September': 0,
        'October': 0,
        'November': 0,
    }

    # for df['intv_actv'] where activation_month is X, get average value
    for month in list(month_time_dict.keys()):
        temp_df = df[df['activation_month'] == month]
        month_time_dict[month] = float(temp_df['interval - activate'].mean())

    return jsonify({'modelRetention': month_time_dict}), 200

