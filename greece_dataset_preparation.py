# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:36:49 2023

@author: jimak
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from sklearn import preprocessing
import scipy
from tabulate import tabulate
from matplotlib import pyplot as pp
import datetime
import statistics
import math
import xgboost as xg
from math import log,sqrt
from scipy import stats
from mlxtend.evaluate import bias_variance_decomp

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD,Adam
from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import soft_dtw

import seaborn as sns
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from keras.callbacks import TensorBoard

from statsmodels.tsa import stattools
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import product
from scipy.optimize import differential_evolution
from dtw import dtw,accelerated_dtw
from scipy import signal

#METHODS
#def lagfeatures(data,featurename, nlags):
#    for i in range(nlags):
#        shiftpos = i+1
#        data[str(featurename)+'lag_'+str(shiftpos)] = data[featurename].shift(shiftpos)
#    return data

def lagfeatures(data,featurename, nlags):
    if nlags==0:
        data=data[featurename]
    else:
        for i in range(nlags):
            shiftpos = i+1
            data[str(featurename)+'lag_'+str(shiftpos)] = data[featurename].shift(shiftpos)
    
        #KEEP COLUMNS THAT START WITH VARIABLE NAME
        data=data[data.columns[pd.Series(data.columns).str.startswith(featurename)]]
    return data

def data_features(df_dataset, dtstr,loadstr):
    
    #Convert to appropriate datetime format
    df_dataset['Datetime'] = pd.to_datetime(df_dataset[dtstr])
    
    #Create time-related features
    df_dataset['Date']=df_dataset.index
    df_dataset['hour'] = df_dataset['Datetime'].dt.hour
    df_dataset['year'] = df_dataset['Datetime'].dt.year
    df_dataset['month'] = df_dataset['Datetime'].dt.month
    df_dataset['quarter'] = df_dataset['Datetime'].dt.quarter
    df_dataset['day'] = df_dataset['Datetime'].dt.day
    df_dataset['dayofweek_num']=df_dataset['Datetime'].dt.dayofweek
    df_dataset['dayofyear'] = df_dataset['Datetime'].dt.dayofyear
    #df_dataset['dayofweek_name']=df_dataset['Datetime'].dt.day_name()
    
    print(df_dataset['Datetime'].dtype)
    print(df_dataset['year'])

    #Create Lag Features for the previous week
    #df_dataset = lagfeatures(df_dataset, featurename='PJME_MW', nlags=168)
    
    #Rolling/expanding features
    df_dataset['Rolling_Mean'] = df_dataset[loadstr].rolling(window=24).mean()
    df_dataset['Expanding_Mean'] = df_dataset[loadstr].expanding(24).mean()
    
    return df_dataset

def data_corrs(df_dataset):
    fig_acf_price_train=plot_acf(df_dataset['Consumption_MW'], lags=24, title= 'Autocorrelation MW')
    fig_pacf_price_train=plot_pacf(df_dataset['Consumption_MW'], lags=24, title ='Partial Autocorrelation MW' )

def prep_in_out(df_dataset):
    #remove null records
    df_dataset = df_dataset.dropna()
    df_dataset =df_dataset.drop(columns=['Datetime'])
    #DEFINE X and y
    X=df_dataset.drop(columns=['Consumption_MW']).values
    y=df_dataset['Consumption_MW'].values
    print(X)
    print(X.shape)
    print(y)
    print(y.shape)
    
    #split dataset to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def calculate_mape(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

#REGION CONSUMPTION DATASET PREPARATION
pd.set_option('display.max_rows',100)
hourly_res_dataset = "time_series_60min_singleindex.csv"
path = "D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\\"+hourly_res_dataset

dataset=pd.read_csv(path) # Load dataset 
dataset.info()

region_id = "GR_" #FOR GREECE
region_flist = dataset.columns[dataset.columns.str.contains(region_id)].values.tolist()
region_flist.append("utc_timestamp")
print(region_flist)
dataset = dataset[region_flist]
#dataset = lagfeatures(dataset, "DE_load_actual_entsoe_transparency", 24)

gr_feature_list = ['GR_load_actual_entsoe_transparency', 'GR_solar_generation_actual', 'GR_wind_onshore_generation_actual']
gr_dataset = dataset[gr_feature_list].copy()

print(gr_dataset)
print(gr_dataset.info())
#print(gr_dataset['utc_timestamp'])
#exit()
gr_dataset = gr_dataset.dropna()

#CREATE NEW DATASET WITH 100 TIMESTEPS TOTAL FOR EACH VARIABLE
fnames = gr_dataset.columns.values.tolist()
print(fnames)
gr_df = pd.DataFrame()
for f in fnames:
    data_tr = lagfeatures(gr_dataset.copy(),f, 99)
    #print (data_tr,data_tr.shape)
    gr_df = pd.concat([gr_df, data_tr], axis=1)
gr_dataset = gr_df
gr_dataset = gr_dataset.dropna()

print(gr_dataset)
print(gr_dataset.shape)
#gr_dataset.to_csv(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\gr_dataset_100t.csv',index=False)


df_names = ['Electricity Load','Solar Generation','Wind Onshore Generation']
k=0
for f in gr_feature_list:
    pp.plot(gr_dataset[f])
    pp.xlabel("Timesteps")
    pp.ylabel(df_names[k]+" (MW)")
    #pp.title("Average Chain Similarity On 168-Hour Subsequences")
    #pp.legend()
    pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\dataset_feature_'+str(k)+'.png', dpi=1000)
    pp.show()
    k=k+1

for f in gr_feature_list:
    fig_acf_price_train=plot_acf(gr_dataset[f], lags=100, title= 'Autocorrelation of '+f)
    fig_pacf_price_train=plot_pacf(gr_dataset[f], lags=100, title ='Partial Autocorrelation of '+f )

exit()


print(gr_dataset.head(1))
for c in gr_dataset.columns:
    print(c)
    nansum = gr_dataset[c].isna().sum()
    print(gr_dataset[c].isna().sum())

exit()

#EXTEND DATASET WITH 24H-LAGGED FEATURES OF EACH VARIABLE
fnames = dataset.columns.values.tolist()
print(fnames)
df_tr = pd.DataFrame()
for f in fnames:
    data_tr = lagfeatures(dataset.copy(),f, 24)
    #print (data_tr,data_tr.shape)
    df_tr = pd.concat([df_tr, data_tr], axis=1)
dataset = df_tr
dataset = data_features(dataset,"utc_timestamp","DE_load_actual_entsoe_transparency")

#drop first 24 rows
dataset = dataset.iloc[25:]
print(dataset,dataset.shape)

"""
extra_data = ['DE_LU_load_actual_entsoe_transparency', 'DE_LU_load_forecast_entsoe_transparency', 'DE_LU_price_day_ahead', 'DE_LU_solar_generation_actual', 'DE_LU_wind_generation_actual', 'DE_LU_wind_offshore_generation_actual', 'DE_LU_wind_onshore_generation_actual']
for e in extra_data:
    pp.plot(dataset[e])
    pp.show()

extra_frame = dataset.copy()
extra_frame = extra_frame.iloc[32852:]
for c in extra_frame.columns:
    print(c)
    nansum = extra_frame[c].isna().sum()
    print(extra_frame[c].isna().sum())
    
    if nansum!=0:
        extra_frame[c] = extra_frame[c].interpolate()
    if nansum>30000:
        extra_frame = extra_frame.drop(columns=[c])
        #sup_list.append(c)

extra_frame.to_csv(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\extra_LU_dataset.csv',index=False)
"""

sup_list=[]
for c in dataset.columns:
    print(c)
    nansum = dataset[c].isna().sum()
    print(dataset[c].isna().sum())
    
    if nansum!=0:
        dataset[c] = dataset[c].interpolate()
    if nansum>30000:
        dataset = dataset.drop(columns=[c])
        sup_list.append(c)
        

print(dataset)
print(sup_list)
#pp.plot(dataset['DE_LU_solar_generation_actual'])
#pp.show()
print("DATASET AFTER PREPARATION")

for c in dataset.columns:
    print(c)
    nansum = dataset[c].isna().sum()
    print(dataset[c].isna().sum())
    if nansum!=0:
        dataset[c] = dataset[c].dropna()

#dataset.to_csv(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\region_dataset.csv',index=False)
dataset.to_csv(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\region_dataset_extended.csv',index=False)


#pp.plot(region_data['DE_load_actual_entsoe_transparency'])
#pp.show()

#pp.plot(region_data["DE_transnetbw_wind_onshore_generation_actual"])
#pp.show()