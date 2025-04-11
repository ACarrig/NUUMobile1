import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from ollama import generate

directory = './backend/userfiles/'  # Path to user files folder

# Function to get the app usage with the values
def get_app_usage(file, sheet):
    file_path = os.path.join(directory, file)

    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)

    app_usage_total = {}
    
    # Check if 'App Usage (s)' column exists before processing
    if "App Usage (s)" in df.columns:
        for usage_str in df["App Usage (s)"].dropna():
            usage_str = str(usage_str)
            apps = usage_str.split(", ")
            for app in apps:
                try:
                    name, time_str = app.rsplit(" ", 1)
                    time = int(time_str.replace("s", ""))

                    if name in app_usage_total:
                        app_usage_total[name] += time
                    else:
                        app_usage_total[name] = time
                except ValueError:
                    continue  # Skip bad formatted data

        return {"app_frequency": app_usage_total}  # Return frequency dictionary
    else:
        return {"app_frequency": {}}  # Return an empty dictionary if column is missing

# Function to get the most used app count across all phones
def get_most_used_app_counts(file, sheet):
    file_path = os.path.join(directory, file)
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)

    most_used_counts = {}

    if "App Usage (s)" in df.columns:
        for usage_str in df["App Usage (s)"].dropna():
            usage_str = str(usage_str)
            apps = usage_str.split(", ")

            usage_dict = {}

            for app in apps:
                try:
                    name, time_str = app.rsplit(" ", 1)
                    time = int(time_str.replace("s", ""))
                    usage_dict[name] = time
                except ValueError:
                    continue  # Bad formatted data

            if usage_dict:
                # Find the most used app in this record
                most_used_app = max(usage_dict, key=usage_dict.get)
                if most_used_app in most_used_counts:
                    most_used_counts[most_used_app] += 1
                else:
                    most_used_counts[most_used_app] = 1

    return {"most_used_counts": most_used_counts}

# Helper method to get a summary from locally running ai model about data
def app_ai_summary(file, sheet):
    # Load the Excel file and sheet
    file_path = os.path.join(directory, file)
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet)

    app_usage_data = get_app_usage(file, sheet)

    # Prepare the prompt for the AI model
    prompt = "Briefly summarize this data, noting features about " \
    "the type of apps and time used for them" + str(app_usage_data) + " " \
    "Avoid using numbers as much as you can and keep your response short"
    
    # Call the AI model to get the summary
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    MODEL_NAME = "llama3.2:1b"
    model_response = generate(MODEL_NAME, prompt)  # Assuming generate handles the API call to the model
    
    # Extract the summary from the model's response
    ai_sum = model_response.get('response', 'No summary available')
    
    return jsonify({'aiSummary': ai_sum})

# # Function to return info about customer app usage
# # This function reads in files as a dataframe one at a time and analyzes them
# # Currently only working for 1 file at a time (b/c returning after doing analysis on first valid file)
# def app_usage_info():
#     directory = os.fsencode('./backend/userfiles/')
#     for file in os.listdir(directory):
#         filename = os.fsdecode(file)
#         if filename.endswith(".xls"): 
#             xls_file = pd.ExcelFile(filename)
#             sheet_names = xls_file.sheet_names

#             # list of lists of app usage 
#             # (each 'valid' sheet has 1 list of app usage containing data for each customer entry)
#             app_usage_lists = []

#             # go over each sheet, for sheets w/ an App Usage (s) column, analyze
#             for sheet in sheet_names:
#                 df = pd.read_excel(xls_file, sheet_name=sheet)
#                 columns = df.columns
#                 if 'App Usage (s)' not in columns:
#                     continue
#                 else:
#                     list = df['App Usage (s)'].tolist()
#                     app_usage_lists.append(list)

#             # app_usage_lists has a dataframes of sheets with needed column
#             #     operate on all valid dataframes
#             return(app_usage_analysis(app_usage_lists))

#         else:
#             continue

# # Helper for app_usage_info
# def app_usage_analysis(app_usage_lists):
#     # go over app_usage_lists and combine them into 1 big list
#     customer_apps_used = []
#     for cur_list in app_usage_lists:
#         customer_apps_used += cur_list

#     # We can see the 5 most used apps on each device
#     # but currently it's a list of strings (not seperated by app)
#     # Reformat customer_apps_used so that each row is a dictionary containing the app and seconds used
#     i = 0
#     for app_time_list in customer_apps_used:
#         # current format: ['Whats app 500s, Tiktok 30000s, Youtube 124s, App4 100s, App5 12s', '', '']
#         # may be a 0 in this column if no data for the device
#         if app_time_list == 0:
#             customer_apps_used[i] = {'none': 0}
#             i += 1
#             continue

#         # first split by comma to seperate out each app
#         apps_list = app_time_list.split(", ")

#         # format of apps_list: [['Whats app 500s', 'Tiktok 30000s', Youtube 124s'], [], []]
#         # split each app into a name and use time, and add to correct list
#         app_names = []
#         app_times = []
#         for app in apps_list:
#             app_info = app.split(" ")
#             app_names.append(app_info[0])
#             app_times.append(int(app_info[1][:-1]))

#         # zip together the names and times lists & create a dictionary
#         apps = dict(zip(app_names, app_times))
    
#         # each customer row now holds a dictionary of their 5 most used apps and the times
#         customer_apps_used[i] = apps
#         i += 1

#     ###########################################################################
#     # Create a dataframe for the apps and their usage, default val = 0
#     app_df = pd.DataFrame(customer_apps_used)
#     app_df = app_df.fillna(0).astype(int)
#     ###########################################################################
    
#     # Figure out total usage time by app, most used apps, 
#     #    most common most used app (this may help remove outliers who used one app for a very long time)

#     # Total usage time by app
#     apps = list(app_df.columns[:-1]) # removing 'none'
#     app_sums_hrs = []
#     for app in apps:
#         app_sums_hrs.append(app_df[f'{app}'].sum() / 3600) # sum apps from it's col vals
    
#     rounded_hrs = [round(num, 2) for num in app_sums_hrs]
#     app_usage_dict = dict(zip(apps, rounded_hrs)) # dict of {'app': time <hrs used>}

#     # Most common most used app
#     # list of the column name pertaining to max value in each row
#     max_values_colname = app_df.idxmax(axis=1) 
#     common_fav_apps_dict = {count: 0 for count in apps} # dict of {apps: <name> count: <# times most used>}
#     for app_name in max_values_colname:
#         common_fav_apps_dict[app_name] += 1

#     return jsonify({'hours': app_usage_dict, 'favorite': common_fav_apps_dict}), 200
