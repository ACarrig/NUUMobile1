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

    # get total num of defects
    num_defects = 0
    for value in defect_counts.values():
        num_defects += value

    

    return

# generate an ai summary about the device returns for a particular file
def returns_summary(file, sheet):
    return