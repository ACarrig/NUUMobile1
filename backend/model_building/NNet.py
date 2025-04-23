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

def_cols = ['Churn', 'interval date', 'last boot date', 'activate date',
       'last boot - activate', 'last boot - interval', 'Age Range',
       'Model_A10L', 'Model_A11L', 'Model_A15', 'Model_A23 ', 'Model_A23 PLus',
       'Model_A23 Plus', 'Model_A25', 'Model_A9L', 'Model_B10', 'Model_B15',
       'Model_B20', 'Model_B20TPU', 'Model_B30', 'Model_B30 ', 'Model_B30 Pro',
       'Model_Earbuds A', 'Model_Earbuds B', 'Model_F4L', 'Model_G5',
       'Model_N10', 'Model_Tab 8 Plus', 'Model_Tab10', 'Model_X6P',
       'Model_nan', 'Warranty_No', 'Warranty_Yes', 'Warranty_nan',
       'Promotion Email_0.0', 'Promotion Email_1.0', 'Promotion Email_nan',
       'Registered Email_0.0', 'Registered Email_1.0', 'Registered Email_nan',
       'Sale Channel_B2C 3rd party', 'Sale Channel_B2C NUU Website',
       'Sale Channel_nan', 'Slot 1_AT&T — DIGICEL', 'Slot 1_AT&T — LIBERTY',
       'Slot 1_Assurance Wireless', 'Slot 1_CC Network', 'Slot 1_CMCC',
       'Slot 1_CU', 'Slot 1_Emergency calls only',
       'Slot 1_Emergency calls only — T-Mobile', 'Slot 1_HOME', 'Slot 1_JIO',
       'Slot 1_Lebara', 'Slot 1_Metro by T-Mobile', 'Slot 1_No service',
       'Slot 1_Sin servicio', 'Slot 1_Solo llamadas de emergencia',
       'Slot 1_T-Mobile', 'Slot 1_T-Mobile Wi-Fi Calling', 'Slot 1_UNICOM',
       'Slot 1_Verizon', 'Slot 1_Visible', 'Slot 1_Vodafone', 'Slot 1_airtel',
       'Slot 1_cricket', 'Slot 1_uninserted', 'Slot 1_只能拨打紧急呼救电话',
       'Slot 1_nan', 'Slot 2_AT&T — airtel', 'Slot 2_CMCC', 'Slot 2_CU',
       'Slot 2_Emergency calls only', 'Slot 2_No service', 'Slot 2_T-Mobile',
       'Slot 2_UNICOM', 'Slot 2_Vodafone', 'Slot 2_uninserted', 'Slot 2_nan']

def_args = {'hidden_layer_sizes':[50]*50}
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
        #processing data
        self.Default_Process(sheetPath, sheetNames, pre=False)
        self._encode()
        #init neural net with given arguments
        self.narg=N_args
        self.neural_net =  MLPClassifier(**N_args)
        self._Split()
        self.neural_net.fit(self.X_train,self.Y_train)
    
    
    def Sheet_Predict_default(self, sp, sn):
        dt = pd.read_excel(sp, sheet_name=sn)
        if ('Office Date' in dt.columns):
            self.data = self._ProcessD1(dt)
        else:
            self.data = self._ProcessD2(dt)
        for col in ['interval date', 'last boot date', 'activate date', 'last boot - activate', 'last boot - interval']:
            self.data[col] = pd.to_datetime(self.data[col], errors='coerce').apply(Churn_Network._con_day)
        for col in self.data.columns:
            self.data[col] = self.data[col].apply(self._CN)
        self.data = self._encode()
        self.fix()
        self._Split()
        return self.predict(self.X)
    
    def Default_Test(self, CV_Scoring='balanced_accuracy'):
        self.cv_scores =  cross_val_score(self.neural_net, self.X, self.Y, cv=5, scoring=CV_Scoring)
        print(self.report())
        
    def Default_Process(self, sheetPath, sheetnames, pre=True):
        df01 = pd.read_excel(sheetPath, sheet_name=sheetnames[0])
        df02 = pd.read_excel(sheetPath, sheet_name=sheetnames[1])
        self.simple_process(df01, df02)
        self.ued= self.data
        if (not pre):
            self.clean_churn()
        
        
    def report(self):
        pred = self.neural_net.predict(self.X_test)
        return classification_report(self.Y_test, pred)
    
    def fix(self):
        for x in def_cols:
            if (not x in self.data.columns):
                self.data[x] = [0] * len(self.data)
        self.data = self.data[def_cols]
        
        
    def predict(self, data):
        return self.neural_net.predict(data)
    
    def save_model(self, path):
        with open(path, "wb") as file:
            pickle.dump((self.neural_net), file)
    
    def load_model(self, path):
        with open(path, "rb") as file:
            self.neural_net = pickle.load(file)
    

    #going to return a dictionary with feature importance and update self.lip
    def LOO_Feature_Importance(self, verbose=False):
        unprocessed_data = self.data.clone()
        df = unprocessed_data
        bac = self.Default_Test()
        accuracies = {}
        reports = {}

        n = 1
        t = len(unprocessed_data.columns)-1
        for item in unprocessed_data.columns:
            if item!="Churn":
                pd = df.drop(columns=[item])
                pdd = self._encode(gone=[item])
                y = pdd["Churn"]
                x = pdd.drop(columns=["Churn"])
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
                NN =  MLPClassifier(hidden_layer_sizes=self.args)
                NN.fit(X_train,y_train)
                y_pred = NN.predict(X_test)
                accuracies[item] = bac - balanced_accuracy_score(y_test, y_pred)
                reports[item] = classification_report(y_test, y_pred)
                if verbose: print(n, '/', t)
                n+=1
        self.lip = accuracies
        return (accuracies, reports)
    #going to return a dictionary with feature importance and update self.pip, NYI
    def PER_Feature_Importance(self):
        self.pip = permutation_importance(self.neural_net, self.X, self.Y, scoring="balanced_accuracy").importances_mean
        return self.pip
    
    def _ProcessD1(self, d1):
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
        
        cList1 = []
        for ind in range(d1s):
            cList1.append(Churn_Network._classify(d1n["last boot - activate"][ind], cF=d1n["Churn"][ind]))
        d1n["Churn"] = cList1
        d1n["Age Range"] = [0]*len(d1n)
        return d1n
    
    def _ProcessD2(self, d2):
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
        cList2 = [] 
        d2s = len(d2n)
        for ind in range(d2s):
            cList2.append(Churn_Network._classify(d2n["last boot - activate"][ind], category=d2n["Type"][ind]))
        d2n["Churn"] = cList2
        d2n.drop(columns = ["Number of Sim", "Sim Country", 
                            "Screen Usage (s)", "Bluetooth (# of pairs)",
                            "Wifi/Internet Connection", "Type", "Sim Card"], inplace=True)
        d2n.rename(columns={"last bootl date":"last boot date"}, inplace=True)
        
        #d2n["Promotion Email"] = [None] * d2s
        d2n["Age Range"] = d2n["Age Range"].apply(Churn_Network._conv_AGE)
        d2n["Promotion Email"] = [None] * len(d2n)
        return d2n
        
    def simple_process(self, d1, d2):
        self.data =pd.concat([self._ProcessD1(d1),self._ProcessD2(d2)], ignore_index=True)
        #print(df.columns)
        for col in ['interval date', 'last boot date', 'activate date', 'last boot - activate', 'last boot - interval']:
            self.data[col] = pd.to_datetime(self.data[col], errors='coerce').apply(Churn_Network._con_day)
        for col in self.data.columns:
            self.data[col] = self.data[col].apply(self._CN)
        
        
    
    def clean_churn(self):
        rList = []
        for ind in range(len(self.data)):
            if (self.data["Churn"][ind]==-1):
                rList.append(ind)
        self.data = self.data.drop(rList, axis = 0)
        
   
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
    
    def _encode(self, remove=[]):
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
        dfm = pd.get_dummies(self.data, columns = cl, drop_first=True, dummy_na=True)
        self.data = dfm
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



'''
def main():
    cn = Churn_Network(init_mode='default_model_train', args=def_all)
    return cn
c = main()
'''
'''
Index(['Model', 'Warranty', 'Churn', 'Promotion Email', 'Registered Email',
       'interval date', 'last boot date', 'activate date',
       'last boot - activate', 'last boot - interval', 'Sale Channel',
       'Slot 1', 'Slot 2'],
      dtype='object')


Index(['Sale Channel', 'Model', 'Warranty', 'Slot 1', 'Slot 2',
       'Registered Email', 'last boot - activate', 'last boot - interval',
       'interval date', 'last boot date', 'activate date', 'Age Range',
       'Churn'],
      dtype='object')

'''