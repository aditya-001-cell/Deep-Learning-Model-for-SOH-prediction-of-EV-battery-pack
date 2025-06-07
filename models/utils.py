import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Bidirectional, Dropout, Dense, Input

def dataCleaning(training_data):
        SEED = 42
        np.random.seed(SEED)
        random.seed(SEED)
        tf.random.set_seed(SEED)
        Y_actual=[]

        def preprocess_dataset(df):
            """Preprocesses dataset by normalizing and scaling features."""
            feature = df.columns.tolist()[:-1]
            source = df.copy()
            Y_actual.append(source['SOH'][timesteps:].values)
            scaler1=StandardScaler()
            exclude_features = ['SOH']
            general_features = [col for col in feature if col not in exclude_features]
            source[general_features] = scaler1.fit_transform(source[general_features])

            scaler2 = StandardScaler()
            source[['SOH']] = scaler2.fit_transform(source[['SOH']])

            return source,scaler2

        def create_dataset(X, Y, timesteps,scaler_SOH):
            """Creates time series dataset with specified timesteps."""
            Xs, Ys = [], []
            scale=[]
            for i in range(len(X) - timesteps):
                Xs.append(X.iloc[i:i+timesteps].values)
                Ys.append(Y.iloc[i + timesteps])
                scale.append(scaler_SOH)
            return np.array(Xs), np.array(Ys),np.array(scale)


        timesteps = 20
        X_combined, Y_combined = [], []
        scalling=[]

        for dataset in training_data:
            processed_data,scaler_SOH = preprocess_dataset(dataset)
            X, Y ,S= create_dataset(processed_data.iloc[:, :-1], processed_data.SOH, timesteps,scaler_SOH)
            X_combined.append(X)
            Y_combined.append(Y)
            scalling.append(S)


        X_combined = np.vstack(X_combined)
        Y_combined = np.hstack(Y_combined)
        scalling=np.hstack(scalling)
        Y_actual=np.concatenate(Y_actual)

        shuffle_indices = np.arange(X_combined.shape[0])
        np.random.shuffle(shuffle_indices)

        X_combined = X_combined[shuffle_indices]
        Y_combined = Y_combined[shuffle_indices]
        scalling=scalling[shuffle_indices]
        Y_actual=Y_actual[shuffle_indices]
        
        return X_combined,Y_combined,scalling,Y_actual
        print("Dataset prepared with consistent shuffling.")

def dataImputation(data):
    for idx in range(len(data)):
        df = data[idx]
        def replace_inf_with_mean(df, column):
            column_mean = df.loc[~df[column].isin([np.inf, -np.inf]), column].mean()
            df[column] = df[column].replace([np.inf, -np.inf], column_mean)

        general_features = df.columns.tolist()[:-1]
        for col in general_features:
            if df[col].isin([np.inf, -np.inf]).any():
                replace_inf_with_mean(df, col)

        data[idx] = df.copy()
    return data
     