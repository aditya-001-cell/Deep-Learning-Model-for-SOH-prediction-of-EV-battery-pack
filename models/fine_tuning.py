
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Bidirectional, Dropout, Dense, Input
from utils import dataCleaning
from utils import dataImputation
import os
from tensorflow.keras.models import load_model

class ModelFinetuning:
    def __init__(self):
        pass
    

    def Model(self,timesteps,feature_size):
        input_layer = Input(shape=(timesteps, feature_size))
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',trainable=False)(input_layer)
        lstm1 = Bidirectional(LSTM(100, return_sequences=True,trainable=False))(conv1)
        dropout1 = Dropout(0.2)(lstm1)
        lstm2 = Bidirectional(LSTM(64,return_sequences=True))(dropout1)
        dropout2 = Dropout(0.2)(lstm2)
        lstm3=Bidirectional(LSTM(32))(dropout2)
        dropout3 = Dropout(0.2)(lstm3)
        dense1 = Dense(16, activation='relu')(dropout3)
        dropout4 = Dropout(0.2)(dense1)
        output_layer = Dense(1)(dropout4)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model
    
    def fine_tuning(self,timesteps,feature_size,X,Y,model_name):
        model = self.Model(timesteps,feature_size)
        model.build(input_shape=(None, timesteps, feature_size))
        file_path=os.path.join("models","saved_models","base_model.weights.h5")
        model.load_weights(file_path)
        history = model.fit(X, Y, epochs=50, batch_size=32, verbose=2)
        model_path='models/saved_models/'+model_name+'_fine_tuned_model.keras'
        model.save(model_path)

    def testResults(self,test_data,model):
        SEED = 42
        np.random.seed(SEED)
        random.seed(SEED)
        tf.random.set_seed(SEED)
        for df in test_data:
            dataset=df
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

            processed_data,scaler_SOH = preprocess_dataset(dataset)
            X, Y ,S= create_dataset(processed_data.iloc[:, :-1], processed_data.SOH, timesteps,scaler_SOH)
            X_combined.append(X)
            Y_combined.append(Y)
            scalling.append(S)

            X_combined = np.vstack(X_combined)
            Y_combined = np.hstack(Y_combined)
            scalling=np.hstack(scalling)
            Y_actual=np.concatenate(Y_actual)

            prediction=model.predict(X_combined)
            prediction=prediction.flatten()
            temp=[]
            for i in range(len(prediction)):
                a = scalling[i].inverse_transform(prediction[i].reshape(-1, 1))
                temp.append(a)

            prediction=temp
            prediction = [x.item() for x in prediction]
            error=0
            for i in range(len(Y_actual)):
                error+=abs(prediction[i]-Y_actual[i])
            print("Error: ",error/len(Y_actual))

            rmse = np.sqrt(np.mean((Y_actual - prediction) ** 2))
            print("RMSE:", rmse)

            plt.plot(Y_actual,color='g')
            plt.plot(prediction,color='r')
        plt.show()
      
    def testing(self, test_data, timesteps, feature_size, model_name):
        saved_model_name = model_name + "_fine_tuned_model.keras"
        base_dir = os.path.dirname(os.path.abspath(__file__))  
        file_path = os.path.join(base_dir, "saved_models", saved_model_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at: {file_path}")
        model = load_model(file_path)
        self.testResults(test_data, model)
    





