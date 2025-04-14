import pandas as pd
import numpy as np
import json
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from ast import literal_eval
from sklearn.inspection import permutation_importance

df01 = pd.read_excel("UW_Churn_Pred_Data.xls", sheet_name="Data Before Feb 13")
df02 = pd.read_excel("UW_Churn_Pred_Data.xls", sheet_name="Data")

'''
Private function which classifies a data point for training
'''
def classify(lbma, cF=None, category=None):
    if(cF==1 or category=="Return"):
        return 1
    if(isinstance(lbma, pd.Timedelta)):
        lm = lbma.days
    elif(isinstance(lbma, int)):
        lm = lbma
    else:
        lm=0
    if(lm>=30 or cF==0 or category=="Repair"):
        return 0
    return -1

#Credit Ik Teng
def convert_arabic_numbers(text):
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"
    return text.translate(str.maketrans(arabic_digits, western_digits)) if isinstance(text, str) else text

'''
Helper class used to organize model use

'''

class Network:
    def __init(self):
        pass
    def loadModel(self):
        pass
    def trainModel(self):
        pass
    


def convert_source(source):
    if (source=="B2C Amazon"):
        return "B2C 3rd party"
    return source

def conv_AGE(st):
    if (isinstance(st, str)):
        return int(st[0])
    else:
        return None
def con_day(date):
    return date.day
def combine(d1, d2):
    d1n = d1.drop(columns=['Device number', 'Month','Office Date', 
                       'Office Time In', 'Type', 
                       'Final Status', 'Defect / Damage type', 
                       'Responsible Party'])

    # Convert date columns (Credit Ik Teng)
    for col in ['last_boot_date', 'interval_date', 'active_date']:
        d1n[col] = d1n[col].astype(str).apply(convert_arabic_numbers)
        d1n[col] = pd.to_datetime(d1n[col], errors='coerce').apply(con_day)
    d1n["last boot - activate"] = d1n["last_boot_date"] - d1n["active_date"]
    d1n["last boot - interval"] = d1n["last_boot_date"] - d1n["interval_date"]
    d1s = len(d1n)
    d1n["Sale Channel"] = d1n["Source"].apply(convert_source)
    
    d1n.drop(columns=['Source'], inplace=True)
    
    s1l = ['uninserted'] * len(d1n)
    s2l = ['uninserted'] * len(d1n)
    for i in range(len(d1n)):
        slots = d1n['sim_info'][i]
        if isinstance(slots, str):
            if (slots[0]=='['):
                ps = literal_eval(slots.replace('\r', ''))
                for item in ps:
                    if item["slot_index"] == 0:
                        s1l[i] = item["carrier_name"]
                    else:
                        s2l[i] = item["carrier_name"]
            elif(slots!='uninserted'):
                s1l[i] = None
                s2l[i] = None
        else:
            s1l[i] = None
            s2l[i] = None
    d1n['Slot 1'] = s1l
    d1n['Slot 2'] = s2l
    d1n.drop(columns=['sim_info'], inplace=True)
 
    d1n.rename(inplace=True, columns={"Product/Model #":"Model", "promotion_email":"Promotion Email", "register_email":"Registered Email", 
                                      "interval_date":"interval date", "last_boot_date":"last boot date", "active_date":"activate date"})
    #d1n["Age Range"] = [None] * d1s
    d2n = d2.drop(columns=['Feedback', 'Verification', 'Defect / Damage type', 
                       'Responsible Party', 'Spare Parts Used if returned', 'Final Status', 
                       'App Usage (s)', 'Wallpaper', 'Customer Service Requested'])
    s1l = ['uninserted'] * len(d2n)
    s2l = ['uninserted'] * len(d2n)
    for i in range(len(d2n)):
        slots = d2n['Sim Card'][i]
        if isinstance(slots, str):
            if (slots[0]=='['):
                ps = literal_eval(slots.replace('\r', ''))
                for item in ps:
                    itemp = literal_eval(item)
                    if itemp["slot_index"] == 0:
                        s1l[i] = itemp["carrier_name"]
                    else:
                        s2l[i] = itemp["carrier_name"]
            elif(slots!='uninserted'):
                s1l[i] = None
                s2l[i] = None
        else:
            s1l[i] = None
            s2l[i] = None
    
    d2n['Slot 1'] = s1l
    d2n['Slot 2'] = s2l
    
    
    
    cList1 = []
    for ind in range(d1s):
        cList1.append(classify(d1n["last boot - activate"][ind], cF=d1n["Churn"][ind]))
    d1n["Churn"] = cList1
    
    cList2 = [] 
    d2s = len(d2n)
    for ind in range(d2s):
        cList2.append(classify(d1n["last boot - activate"][ind], category=d2n["Type"][ind]))
    d1n["Churn"] = cList1
    d2n["Churn"] = cList2
    d2n.drop(columns = ["Number of Sim", "Sim Country", 
                        "Screen Usage (s)", "Bluetooth (# of pairs)",
                        "Wifi/Internet Connection", "Type", "Sim Card"], inplace=True)
    d2n.rename(columns={"last bootl date":"last boot date"}, inplace=True)
    
    #d2n["Promotion Email"] = [None] * d2s
    d2n["Age Range"] = d2n["Age Range"].apply(conv_AGE)
    #print(d1n.columns,'\n', d2n.columns)
    #print(len(d1n.columns), len(d2n.columns))
    df = pd.concat([d1n, d2n], ignore_index=True)
    #print(df.columns)
    
    for col in ['interval date', 'last boot date', 'activate date', 'last boot - activate', 'last boot - interval']:
        df[col] = pd.to_datetime(df[col], errors='coerce').apply(con_day)
    for col in df.columns:
        df[col] = df[col].apply(CN)
    
    rList = []
    for ind in range(len(df)):
        if (df["Churn"][ind]==-1):
            rList.append(ind)
    df = df.drop(rList, axis = 0)
    
    return df
def CN(x):
    if (x!=x):
        return -1
    else:
        return x

def process_frame(df, gone=[]):
    #encoding
    cl = ["Model", 
          "Warranty", 
          'Promotion Email', 
          'Registered Email',
          'Sale Channel', 
          'Slot 1',
          'Slot 2']
    for item in gone:
        if (item in cl):
            cl.remove(item)
    dfm = pd.get_dummies(df, columns = cl, drop_first=True, dummy_na=True)
    
    return dfm
    
    
def run_sample():
    a = combine(df01, df02)
    df = process_frame(a)
    dfstat = [0,0]
    for item in df["Churn"]:
        dfstat[item]+=1
    
    
    y = df["Churn"]
    x = df.drop(columns=["Churn"])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    
    NNet = MLPClassifier(hidden_layer_sizes=[10]*10)
    
    NNet.fit(X_train,y_train)
    
    cv_scores = cross_val_score(NNet, X_train, y_train, cv=5, scoring='balanced_accuracy')
    print(f"Mean Cross-Validation F1 Score: {cv_scores.mean():.4f}")
    
    
    
    y_pred = NNet.predict(X_test)
    
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred))
    m = [[0,0],[0,0]]
    y_it = y_test.reset_index().drop(columns='index')
    for item in range(len(y_it)):
        m[y_it["Churn"][item]][y_pred[item]]+=1
    
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned (0)', 'Churned (1)'], 
                yticklabels=['Not Churned (0)', 'Churned (1)'])
    
    plt.title('Confusion Matrix for Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    
def Feature_Importance(NetNodes, unprocessed_data):
    df = unprocessed_data
    accuracies = {}
    reports = {}

    n = 1
    t = len(unprocessed_data.columns)-1
    for item in unprocessed_data.columns:
        if item!="Churn":
            pd = df.drop(columns=[item])
            pdd = process_frame(pd, gone=[item])
            y = pdd["Churn"]
            x = pdd.drop(columns=["Churn"])
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            NN =  MLPClassifier(hidden_layer_sizes=NetNodes)
            NN.fit(X_train,y_train)
            y_pred = NN.predict(X_test)
            accuracies[item] = balanced_accuracy_score(y_test, y_pred)
            reports[item] = classification_report(y_test, y_pred)
            print(n, '/', t)
            n+=1
    return (accuracies, reports)
a = combine(df01, df02)#.drop(columns = ['Sale Channel'])
b = Feature_Importance([10]*10, a)

def PlotAcc(accuracies):
    s = 'abcdefghijklmnopqrstuvwxyz'
    a = {}
    l1 = []
    ind = 0
    for item in accuracies.keys():
        a[item] = 1-accuracies[item]
        l1.append(s[ind])
        print(s[ind],  ': ', item)
        ind+=1
    l2 = []
    for t in accuracies.keys():
        l2.append(a[t])
    plt.bar(l1, l2)
    plt.title("Feature Relevance/Change in accuracy without feature")
    plt.xlabel('Column/Feature')
    plt.ylabel('Change in accuracy')
    plt.show()

def pev(cols, reports):
    for item in cols:
        if (item!='Churn'):
            print(item, '\n', reports[item])
pev(a.columns, b[0])
PlotAcc(b[0])