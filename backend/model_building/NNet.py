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
import pickle

'''
Helper methods start here

'''

'''

constructor modes:
    empty(none)
    load_model(filepath)
    default_model_train(sheetPath, sheetNames, N_args)
    
    NYI:
    custom_model_training(dataframe)
'''

def_args = {'hidden_layer_sizes':[10]*10}
def_snames = ["Data Before Feb 13", "Data"]
def_path = "UW_Churn_Pred_Data.xls"
def_all = {'sheetPath':def_path, 'sheetNames':def_snames, 'N_args':def_args}


class Churn_Network:
    def __init__(self, init_mode='empty', args=None):
        self.lip=None
        self.pip=None
        self.narg=None
        self.neural_net=None
        self.cv_scores=None
        self.data=None
        self.ued=None
        self.LOO=None
        self.per=None
        if (init_mode=='empty'.lower() or init_mode==None):
            return
        if (init_mode=='default_model_train'.lower()):
            return self.Default_Train(**args)
        if (init_mode.lower()=='load_model'):
            self.load_model(args)
            
    def Default_Train(self, sheetPath, sheetNames, N_args):
        df01 = pd.read_excel(sheetPath, sheet_name=sheetNames[0])
        df02 = pd.read_excel(sheetPath, sheet_name=sheetNames[1])
        #processing data
        self.data = self._Process_Data(df01, df02)
        self.data = Churn_Network._encode(self.data)
        #init neural net with given arguments
        self.narg=N_args
        self.neural_net =  MLPClassifier(**N_args)
        self._Split()
        self.neural_net.fit(self.X_train,self.Y_train)
        
        
    
    def Default_Test(self, CV_Scoring='balanced_accuracy'):
        self.cv_scores =  cross_val_score(self.neural_net, self.X, self.Y, cv=5, scoring=CV_Scoring)
        print(self.report())
        
    def Default_Process(self, sheetPath, sheetnames):
        df01 = pd.read_excel(sheetPath, sheet_name=sheetnames[0])
        df02 = pd.read_excel(sheetPath, sheet_name=sheetnames[1])
        self.data = self._Process_Data(df01, df02)
        self.ued= self.data
        self.data = Churn_Network._encode(self.data)
        self._Split()
        
        
    def report(self):
        pred = self.neural_net.predict(self.X_test)
        return classification_report(self.Y_test, pred)
    

    def predict(self, data):
        return self.neural_net.predict(data)
    
    def save_model(self, path):
        with open(path, "wb") as file:
            pickle.dump((self.neural_net, self.narg), file)
    
    def load_model(self, path):
        with open(path, "rb") as file:
            self.neural_net, self.narg = pickle.load(file)
    

    #going to return a dictionary with feature importance and update self.lip, NYI
    def LOO_Feature_Importance(self):
        pass
    #going to return a dictionary with feature importance and update self.pip, NYI
    def PER_Feature_Importance(self):
        pass
    
    def _Process_Data(self, d1, d2, evaluating=False):
        d1n = d1.drop(columns=['Device number', 'Month','Office Date', 
                           'Office Time In', 'Type', 
                           'Final Status', 'Defect / Damage type', 
                           'Responsible Party'])

        # Convert date columns (Credit Ik Teng)
        for col in ['last_boot_date', 'interval_date', 'active_date']:
            d1n[col] = d1n[col].astype(str).apply(Churn_Network._convert_arabic_numbers)
            d1n[col] = pd.to_datetime(d1n[col], errors='coerce').apply(Churn_Network._con_day)
        d1n["last boot - activate"] = d1n["last_boot_date"] - d1n["active_date"]
        d1n["last boot - interval"] = d1n["last_boot_date"] - d1n["interval_date"]
        d1s = len(d1n)
        d1n["Sale Channel"] = d1n["Source"].apply(Churn_Network._convert_source)
        
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
            cList1.append(Churn_Network._classify(d1n["last boot - activate"][ind], cF=d1n["Churn"][ind]))
        d1n["Churn"] = cList1
        
        cList2 = [] 
        d2s = len(d2n)
        for ind in range(d2s):
            cList2.append(Churn_Network._classify(d1n["last boot - activate"][ind], category=d2n["Type"][ind]))
        d1n["Churn"] = cList1
        d2n["Churn"] = cList2
        d2n.drop(columns = ["Number of Sim", "Sim Country", 
                            "Screen Usage (s)", "Bluetooth (# of pairs)",
                            "Wifi/Internet Connection", "Type", "Sim Card"], inplace=True)
        d2n.rename(columns={"last bootl date":"last boot date"}, inplace=True)
        
        #d2n["Promotion Email"] = [None] * d2s
        d2n["Age Range"] = d2n["Age Range"].apply(Churn_Network._conv_AGE)
        #print(d1n.columns,'\n', d2n.columns)
        #print(len(d1n.columns), len(d2n.columns))
        df = pd.concat([d1n, d2n], ignore_index=True)
        #print(df.columns)
        
        for col in ['interval date', 'last boot date', 'activate date', 'last boot - activate', 'last boot - interval']:
            df[col] = pd.to_datetime(df[col], errors='coerce').apply(Churn_Network._con_day)
        for col in df.columns:
            df[col] = df[col].apply(self._CN)
        if not training:
            rList = []
            for ind in range(len(df)):
                if (df["Churn"][ind]==-1):
                    rList.append(ind)
            df = df.drop(rList, axis = 0)
        
        return df
    def _Split(self):
        self.X = self.data.drop(columns=["Churn"])
        self.Y = self.data["Churn"]
        self.X_train, self.X_test, self.Y_train, self.Y_test =  train_test_split(self.X, self.Y, test_size=0.2)

    def _classify(lbma, cF=None, category=None):
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
    
    def _convert_arabic_numbers(text):
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        western_digits = "0123456789"
        return text.translate(str.maketrans(arabic_digits, western_digits)) if isinstance(text, str) else text
    
    def _convert_source(source):
        if (source=="B2C Amazon"):
            return "B2C 3rd party"
        return source
    
    def _conv_AGE(st):
        if (isinstance(st, str)):
            return int(st[0])
        else:
            return None
    def _con_day(date):
        return date.day
    
    def _encode(df, remove=[]):
        #encoding
        cl = ["Model", 
              "Warranty", 
              'Promotion Email', 
              'Registered Email',
              'Sale Channel', 
              'Slot 1',
              'Slot 2']
        for item in remove:
            if (item in cl):
                cl.remove(item)
        dfm = pd.get_dummies(df, columns = cl, drop_first=True, dummy_na=True)
        
        return dfm
    def _CN(self, x):
        if (x!=x):
            return -1
        else:
            return x
    
    def _cval_avg(self, args=None):
        out = cross_val_score(self.neural_net, self.X, self.Y, cv=5, scoring='balanced_accuracy')
        return sum(out)/len(out)

    def _get_acc(model, test, actual):
        r = model.predict(test)
        return balanced_accuracy_score(r, actual)


def main():
    cn = Churn_Network(init_mode='default_model_train', args=def_all)
    return cn