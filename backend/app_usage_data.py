import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to return info about customer app usage
def app_usage_info():

    directory = os.fsencode('./backend/userfiles/')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".xls"): 
            xls_file = pd.ExcelFile(filename)
            
        else:
            continue