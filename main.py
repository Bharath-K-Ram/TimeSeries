import numpy as np
import pandas as pd
from HandlingData.dateHandling import date_Handling
from HandlingData.Date_handling_for_index import eda_date
from tslearn.datasets import UCR_UEA_datasets

# Univariate -->
# Air passengers : /home/local/ZOHOCORP/bharath-12053/Desktop/ML/ML_Projects/PythonPractice/AirPassengers.csv
# Wind generation : /home/local/ZOHOCORP/bharath-12053/Desktop/TSA/Time_Series_Datasets-main/Wind Gen.csv


# Multivariate -->
# PVR Cinemas Stock price : /home/local/ZOHOCORP/bharath-12053/Downloads/PVR.csv
# Pump --> /home/local/ZOHOCORP/bharath-12053/Desktop/TSA/dataset/DS/sensor.csv
# Accelerometer --> r'/home/local/ZOHOCORP/bharath-12053/Desktop/TSA/dataset/SKAB-master/data/other/4.csv',sep=';'
# Accelerometer 2 --> r'/home/local/ZOHOCORP/bharath-12053/Desktop/TSA/dataset/SKAB-master/data/valve2/2.csv',sep=';'



if __name__ == '__main__':

    df = pd.read_csv(r'/home/local/ZOHOCORP/bharath-12053/Desktop/TSA/dataset/SKAB-master/data/valve2/2.csv',sep=';')
    print(df)

    eda_date(df)

    # data_loader = UCR_UEA_datasets()
    # X_train, y_train, X_test, y_test = data_loader.load_dataset("Cricket")
    # print( y_train.shape)
    # size=   X_train.shape[2]
    # print(X_train.shape[2])
    # print(X_train)
    #
    # df=pd.DataFrame(X_train.reshape(-1,size))
    # df.index=np.repeat(np.arange(X_train.shape[0]),X_train.shape[1])+1
    # print(df.head(100))
    # df_index = df.loc[df.index == 1]
    # print(df_index)
    # print(df)
    # l = UCR_UEA_datasets().list_multivariate_datasets()
    # print(l)