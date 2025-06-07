import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def fetchXJTUData():
    path='..\\Data\\XJTU data'
    folder=[f for f in os.listdir(path)]
    battery=[path+'\\'+f for f in folder]
    xjtu_data=[[],[],[],[],[],[]]

    all_battery_files=[]
    for i in range(len(battery)):
        print("battery ",i+1)
        folder_path=battery[i]
        all_files=[f for f in os.listdir(folder_path)]
        battery_data_url=[]
        for file in all_files:
            battery_data_url.append(file)
        all_battery_files.append(battery_data_url)
        battery_type=i+1
        nominal_capacity=2.0
        #if(i==2):
        #   nominal_capacity=2.5
        for file in battery_data_url:
            df=pd.read_csv(folder_path+'/'+file)
            if(len(df) <=50):
                continue

            df['Cycle'] = range(1, len(df) + 1)
            for cycle in range(len(df)):
                if(cycle == 0 or cycle == len(df)):
                    continue
                diff=abs(df['capacity'][cycle]-df['capacity'][cycle-1])
                if(abs(diff)>=0.1):
                    df['capacity'][cycle]=(df['capacity'][cycle+1]+df['capacity'][cycle-1])/2
            df['SOH']=df['capacity']/nominal_capacity
            df.drop(columns='capacity',inplace=True)
            xjtu_data[i].append(df)
    
    return xjtu_data


def fetchTJUData():
    battery=['..\\Data\\Dataset_1_NCA_battery','..\\Data\\Dataset_2_NCM_battery','..\\Data\\Dataset_3_NCM_NCA_battery']
    battery_datasets=[]
    tju_data=[[],[],[]]

    all_battery_files=[]
    for i in range(len(battery)):
        print("battery ",i+1)
        folder_path=battery[i]
        all_files=[f for f in os.listdir(folder_path)]
        battery_data_url=[]
        for file in all_files:
            battery_data_url.append(file)
        all_battery_files.append(battery_data_url)
        battery_type=i+1
        nominal_capacity=3.5
        if(i==2):
            nominal_capacity=2.5
        for file in battery_data_url:
            df=pd.read_csv(folder_path+'/'+file)
            if(len(df) <=50):
                continue

            df['Cycle'] = range(1, len(df) + 1)
            for cycle in range(len(df)):
                if(cycle == 0 or cycle == len(df)):
                    continue
                diff=abs(df['capacity'][cycle]-df['capacity'][cycle-1])
                if(abs(diff)>=0.1):
                    df['capacity'][cycle]=(df['capacity'][cycle+1]+df['capacity'][cycle-1])/2
            df['SOH']=df['capacity']/nominal_capacity
            df.drop(columns='capacity',inplace=True)
            tju_data[i].append(df)
    return tju_data

def fetchMITData():
    path='..\\Data\\MIT data'
    folder=[f for f in os.listdir(path)]
    battery=[path+'\\'+f for f in folder]
    mit_data=[]


    all_battery_files=[]
    for i in range(len(battery)):
        print("battery ",i+1)
        folder_path=battery[i]
        all_files=[f for f in os.listdir(folder_path)]
        battery_data_url=[]
        for file in all_files:
            battery_data_url.append(file)
        all_battery_files.append(battery_data_url)
        battery_type=i+1
        nominal_capacity=1.1
        #if(i==2):
        #   nominal_capacity=2.5
        for file in battery_data_url:
            df=pd.read_csv(folder_path+'/'+file)
            if(len(df) <=50):
                continue

            df['Cycle'] = range(1, len(df) + 1)
            for cycle in range(len(df)):
                if(cycle == 0 or cycle == len(df)):
                    continue
                diff=abs(df['capacity'][cycle]-df['capacity'][cycle-1])
                if(abs(diff)>=0.1):
                    df['capacity'][cycle]=(df['capacity'][cycle+1]+df['capacity'][cycle-1])/2
            df['SOH']=df['capacity']/nominal_capacity
            df.drop(columns='capacity',inplace=True)
            mit_data.append(df)

    return mit_data


def fetchHUSTData():
    path='..\\Data\\HUST data'
    #folder=[f for f in os.listdir('/content/drive/MyDrive/SOH/HUST data')]
    battery=['..\\Data\\HUST data']
    hust_data=[]

    all_battery_files=[]
    for i in range(len(battery)):
        print("battery ",i+1)
        folder_path=battery[i]
        all_files=[f for f in os.listdir(folder_path)]
        battery_data_url=[]
        for file in all_files:
            battery_data_url.append(file)
        all_battery_files.append(battery_data_url)
        battery_type=i+1
        nominal_capacity=1.1
        #if(i==2):
        #   nominal_capacity=2.5
        for file in battery_data_url:
            df=pd.read_csv(folder_path+'/'+file)
            if(len(df) <=50):
                continue

            df['Cycle'] = range(1, len(df) + 1)
            for cycle in range(len(df)):
                if(cycle == 0 or cycle == len(df)):
                    continue
                diff=abs(df['capacity'][cycle]-df['capacity'][cycle-1])
                if(abs(diff)>=0.1):
                    df['capacity'][cycle]=(df['capacity'][cycle+1]+df['capacity'][cycle-1])/2
            df['SOH']=df['capacity']/nominal_capacity
            df.drop(columns='capacity',inplace=True)
            hust_data.append(df)
    return hust_data

def getData():
    xjtu_data=fetchXJTUData()
    tju_data=fetchTJUData()
    mit_data=fetchMITData()
    hust_data=fetchHUSTData()
    #XJTU
    pre_train_data_xjtu=[xjtu_data[0][:4],xjtu_data[1][:4],xjtu_data[2][:4],xjtu_data[3][:8],xjtu_data[4][:4],xjtu_data[5][:4]]
    train_data_xjtu=[xjtu_data[0][4:7],xjtu_data[1][4:7],xjtu_data[2][4:7],xjtu_data[3][8:12],xjtu_data[4][4:7],xjtu_data[5][4:7]]
    test_data_xjtu=[xjtu_data[0][7:],xjtu_data[1][7:],xjtu_data[2][7:],xjtu_data[3][12:],xjtu_data[4][7:],xjtu_data[5][7:]]
    
    #TJU
    pre_train_data_tju=[tju_data[0][:28],tju_data[1][:27],tju_data[2][:4]]
    train_data_tju=[tju_data[0][28:48],tju_data[1][27:48],tju_data[2][4:7]]
    test_data_tju=[tju_data[0][48:],tju_data[1][48:],tju_data[2][7:]]

    #MIT
    total_samples = len(mit_data)  # 124
    pretrain_size = int(0.50 * total_samples)  # 62
    train_size = int(0.40 * total_samples)  # 49
    test_size = total_samples - (pretrain_size + train_size)  # Remaining 13

    # Splitting the dataset
    pre_train_data_mit = mit_data[:pretrain_size]
    train_data_mit = mit_data[pretrain_size:pretrain_size + train_size]
    test_data_mit = mit_data[pretrain_size + train_size:]

    #HUST
    total_samples = len(hust_data)  # 124
    pretrain_size = int(0.50 * total_samples)  # 62
    train_size = int(0.40 * total_samples)  # 49
    test_size = total_samples - (pretrain_size + train_size)  # Remaining 13

    # Splitting the dataset
    pre_train_data_hust = hust_data[:pretrain_size]
    train_data_hust = hust_data[pretrain_size:pretrain_size + train_size]
    test_data_hust = hust_data[pretrain_size + train_size:]

    pre_train_data=[]
    train_data=[]
    test_data=[]

    #XJTU
    for i in range(len(pre_train_data_xjtu)):
        for j in range(len(pre_train_data_xjtu[i])):
            pre_train_data.append(pre_train_data_xjtu[i][j])

    for i in range(len(train_data_xjtu)):
        for j in range(len(train_data_xjtu[i])):
            train_data.append(train_data_xjtu[i][j])

    for i in range(len(test_data_xjtu)):
        for j in range(len(test_data_xjtu[i])):
            if not isinstance(test_data_xjtu[i][j], pd.DataFrame):
               print("xjtu, test")
            test_data.append(test_data_xjtu[i][j])

    #TJU
    for i in range(len(pre_train_data_tju)):
        for j in range(len(pre_train_data_tju[i])):
            pre_train_data.append(pre_train_data_tju[i][j])

    for i in range(len(train_data_tju)):
        for j in range(len(train_data_tju[i])):
            train_data.append(train_data_tju[i][j])

    for i in range(len(test_data_tju)):
        for j in range(len(test_data_tju[i])):
            if not isinstance(test_data_tju[i][j], pd.DataFrame):
               print("tju, test")
            test_data.append(test_data_tju[i][j])

    #MIT
    for i in range(len(pre_train_data_mit)):
        pre_train_data.append(pre_train_data_mit[i])

    for i in range(len(train_data_mit)):
        train_data.append(train_data_mit[i])

    for i in range(len(test_data_mit)):
        if not isinstance(test_data_mit[i], pd.DataFrame):
           print("mit, test")
        test_data.append(test_data_mit[i])

    #HUST
    for i in range(len(pre_train_data_hust)):
        pre_train_data.append(pre_train_data_hust[i])

    for i in range(len(train_data_hust)):
        train_data.append(train_data_hust[i])

    for i in range(len(test_data_hust)):
        if not isinstance(test_data_hust[i], pd.DataFrame):
           print("hust, test")
        test_data.append(test_data_hust[i])


    return pre_train_data,train_data,test_data