import os
import re
import pandas as pd
from flask import Flask, jsonify, request
from ollama import generate

# get info about the reasons why customers returned devices for a particular file
def returns_info(file, sheet):
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, sheet_name=sheet)

    

    return

# generate an ai summary about the device returns for a particular file
def returns_summary(file, sheet):
    return