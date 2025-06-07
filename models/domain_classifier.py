# This script includes the model building for base model
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Bidirectional, Dropout, Dense, Input, Flatten
from utils import dataCleaning
from utils import dataImputation
import os
from tensorflow.keras.models import load_model

class Classifier:
    def __init__(self,timesteps=20,feature_size=17):
        self.timesteps=timesteps
        self.feature_size=feature_size

    def Model(self,timesteps,feature_size):
        input_layer = Input(shape=(timesteps, feature_size))
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(conv1)
        flatten_layer = Flatten()(conv2)
        output_layer = Dense(4, activation='softmax')(flatten_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def DataImputation(self,training_data):
        SEED = 42
        np.random.seed(SEED)
        random.seed(SEED)
        tf.random.set_seed(SEED)
        Y_actual=[]
        def preprocess_dataset(df):
            """Preprocesses dataset by normalizing and scaling features."""
            feature = df.columns.tolist()[:-1]
            source = df.copy()
            Y_actual.append(source['battery'][timesteps:].values)
            scaler1=StandardScaler()
            exclude_features = ['battery']
            general_features = [col for col in feature if col not in exclude_features]
            source[general_features] = scaler1.fit_transform(source[general_features])

            return source

        def create_dataset(X, Y, timesteps):
            """Creates time series dataset with specified timesteps."""
            Xs, Ys = [], []
            scale=[]
            for i in range(len(X) - timesteps):
                Xs.append(X.iloc[i:i+timesteps].values)
                Ys.append(Y.iloc[i + timesteps])
            return np.array(Xs), np.array(Ys),np.array(scale)


        timesteps = 20
        X_combined, Y_combined = [], []
        scalling=[]


        for dataset in training_data:
            processed_data = preprocess_dataset(dataset)
            X, Y ,S= create_dataset(processed_data.iloc[:, :-1], processed_data.battery, timesteps)
            X_combined.append(X)
            Y_combined.append(Y)
            #scalling.append(S)


        X_combined = np.vstack(X_combined)
        Y_combined = np.hstack(Y_combined)
        #scalling=np.hstack(scalling)
        Y_actual=np.concatenate(Y_actual)




        shuffle_indices = np.arange(X_combined.shape[0])
        np.random.shuffle(shuffle_indices)

        X_combined = X_combined[shuffle_indices]
        Y_combined = Y_combined[shuffle_indices]
        Y_actual=Y_actual[shuffle_indices]
        return X_combined,Y_combined,Y_actual
        print("Dataset prepared with consistent shuffling.")

    def training(self,training_data, epochs=15, batch_size=32):
        X,Y,Y_actual=self.DataImputation(training_data)
        feature_classification_model= self.Model(self.timesteps, self.feature_size)
        history = feature_classification_model.fit(X, Y, epochs=15, batch_size=32, verbose=2)
        feature_classification_model.save('models/saved_models/feature_classification_model.keras')

    def testResults(self,test_data,model):
        SEED = 42
        np.random.seed(SEED)
        random.seed(SEED)
        tf.random.set_seed(SEED)
        count=0
        print(len(test_data))
        for df in test_data:
            Y_actual=[]
            def preprocess_dataset(df):
                """Preprocesses dataset by normalizing and scaling features."""
                feature = df.columns.tolist()[:-1]
                source = df.copy()
                Y_actual.append(source['battery'][timesteps:].values)
                scaler1=StandardScaler()
                exclude_features = ['battery']
                general_features = [col for col in feature if col not in exclude_features]
                source[general_features] = scaler1.fit_transform(source[general_features])

                return source

            def create_dataset(X, Y, timesteps):
                """Creates time series dataset with specified timesteps."""
                Xs, Ys = [], []
                scale=[]
                for i in range(len(X) - timesteps):
                    Xs.append(X.iloc[i:i+timesteps].values)
                    Ys.append(Y.iloc[i + timesteps])
                return np.array(Xs), np.array(Ys),np.array(scale)


            timesteps = 20
            X_combined, Y_combined = [], []
            scalling=[]



            processed_data = preprocess_dataset(df)
            X, Y ,S= create_dataset(processed_data.iloc[:, :-1], processed_data.battery, timesteps)
            X_combined.append(X)
            Y_combined.append(Y)
            #scalling.append(S)


            X_combined = np.vstack(X_combined)
            Y_combined = np.hstack(Y_combined)
            #scalling=np.hstack(scalling)
            Y_actual=np.concatenate(Y_actual)
            prediction=model.predict(X_combined)
            output=[]
            for i in range(len(prediction)):
                max_prob=0
                max_idx=-1
                for j in range(len(prediction[i])):
                    if(prediction[i][j] > max_prob):
                        max_prob=prediction[i][j]
                        max_idx=j
                output.append(max_idx)

            correct_pred=0
            wrong_pred=0
            for i in range(len(output)):
                if(Y_actual[i] == output[i]):
                  correct_pred+=1
                else:
                  wrong_pred+=1 
          
            print("Test index --> ",count)
            print("Total sample: ",len(output))
            print("Correct prediction : ",correct_pred)
            print("Wrong prediction : ",wrong_pred)
            print("accuracy : ",correct_pred/len(output))
            count+=1
      
    def testing(self, test_data):
        saved_model_name = "feature_classification_model.keras"

        # Compute path relative to fine_tuning.py
        base_dir = os.path.dirname(os.path.abspath(__file__))  # directory of fine_tuning.py
        file_path = os.path.join(base_dir, "saved_models", saved_model_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at: {file_path}")

        model = load_model(file_path)
        self.testResults(test_data, model)
    






