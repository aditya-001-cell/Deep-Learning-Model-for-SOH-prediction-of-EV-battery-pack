import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
#Feature based probability distribution
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os


class CombinedModel:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__)) 
        self.feature_classification_model = load_model(os.path.join(base_dir, "saved_models", 'feature_classification_model.keras'))
        self.model1=load_model(os.path.join(base_dir, "saved_models", 'xjtu_fine_tuned_model.keras'))
        self.model2=load_model(os.path.join(base_dir, "saved_models", 'tju_fine_tuned_model.keras'))
        self.model3=load_model(os.path.join(base_dir, "saved_models", 'mit_fine_tuned_model.keras'))
        self.model4=load_model(os.path.join(base_dir, "saved_models", 'hust_fine_tuned_model.keras'))
    def feature_classification_data(self,test_df):
        Y_actual=[]
        def preprocess_dataset(df):
            """Preprocesses dataset by normalizing and scaling features."""
            feature = df.columns.tolist()[:-1]
            source = df.copy()
            scaler1=StandardScaler()
            exclude_features = ['battery']
            general_features = [col for col in feature if col not in exclude_features]
            source[general_features] = scaler1.fit_transform(source[general_features])

            return source

        def create_dataset(X, timesteps):
            """Creates time series dataset with specified timesteps."""
            Xs= []
            for i in range(len(X) - timesteps):
                Xs.append(X.iloc[i:i+timesteps].values)
            return np.array(Xs)


        timesteps = 20
        X_combined = []
        scalling=[]



        processed_data = preprocess_dataset(test_df)
        X = create_dataset(processed_data.iloc[:, :-1], timesteps)
        X_combined.append(X)


        X_combined = np.vstack(X_combined)
        return X_combined
    
    def domain_data(self,test_df):
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



        processed_data,scaler_SOH = preprocess_dataset(test_df)
        X, Y ,S= create_dataset(processed_data.iloc[:, :-1], processed_data.SOH, timesteps,scaler_SOH)
        X_combined.append(X)
        Y_combined.append(Y)
        scalling.append(S)


        X_combined = np.vstack(X_combined)
        Y_combined = np.hstack(Y_combined)
        scalling=np.hstack(scalling)
        Y_actual=np.concatenate(Y_actual)
        return X_combined,Y_combined,scalling,Y_actual
    
    def getSOH(self,prediction,scalling):
        prediction=prediction.flatten()
        temp=[]
        for i in range(len(prediction)):
            a = scalling[i].inverse_transform(prediction[i].reshape(-1, 1))
            temp.append(a)

        prediction=temp
        SOH = [float(pred[0][0]) for pred in prediction]
        return SOH

    def calculateSOH(self,test_data):
        prob=[]
        for df in test_data:
            test_df=df
            X_combined=self.feature_classification_data(test_df)
            prediction=self.feature_classification_model.predict(X_combined)
            probability=prediction
            prob.append(probability)
            #domain data
            X_combined,Y_combined,scalling,Y_actual=self.domain_data(test_df)

            #Model 1
            prediction=self.model1.predict(X_combined)
            SOH1 =self.getSOH(prediction,scalling)

            #Model 2
            prediction=self.model2.predict(X_combined)
            SOH2 =self.getSOH(prediction,scalling)

            #Model 3
            prediction=self.model3.predict(X_combined)
            SOH3 =self.getSOH(prediction,scalling)

            #model 4
            prediction=self.model4.predict(X_combined)
            SOH4 =self.getSOH(prediction,scalling)

            #calculate SOH
            SOH=[]
            for i in range(X_combined.shape[0]):
                soh=[SOH1[i],SOH2[i],SOH3[i],SOH4[i]]
                sum=0
                for j in range(len(probability[i])):
                    sum+=(probability[i][j]*soh[j])
                SOH.append(sum)
            #calculate error
            mae = mean_absolute_error(Y_actual, SOH)
            rmse = np.sqrt(mean_squared_error(Y_actual, SOH))

            #ploting
            #print("File : ",file)
            print("MAE:", mae)
            print("RMSE:", rmse)
            plt.plot(Y_actual,color='g',label='Actual SOH')
            plt.scatter(np.arange(len(Y_actual)),SOH,color='r',label='Predicted SOH')
            plt.xlabel("Cycle")
            plt.ylabel("SOH")
            plt.legend()
            plt.show()

            plt.plot(Y_actual,Y_actual)
            plt.scatter(Y_actual,SOH,color='r')
            plt.xlabel("Actual SOH")
            plt.ylabel("Predicted SOH")
            plt.show()

            plt.scatter(np.arange(len(SOH)),SOH-Y_actual,color='r')
            plt.plot([0,len(SOH)],[0,0],'k--')
            plt.xlabel("Cycle")
            plt.ylabel("Error")
            plt.show()


