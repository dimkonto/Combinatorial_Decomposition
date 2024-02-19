# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:08:48 2023

@author: jimak
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import stumpy
import emd
import itertools
import numpy as np
import scipy
import csv
import operator
from tabulate import tabulate
from matplotlib import pyplot as pp
from matplotlib.patches import Rectangle, FancyArrowPatch
import datetime
import statistics
import math
import xgboost as xg
from math import log,sqrt
from scipy import stats
from decimal import Decimal
from mlxtend.evaluate import bias_variance_decomp
from hyperspy.signals import Signal1D

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression, HuberRegressor, Lars, Lasso, RANSACRegressor, TheilSenRegressor, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
###FOR CUSTOM LAYER CREATION
from keras import backend as K
from keras.layers import Layer
###
from keras.models import Sequential
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import InputLayer
from keras.layers import TimeDistributed, RepeatVector
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
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
from tslearn.metrics import dtw,soft_dtw

import seaborn as sns
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from keras.callbacks import TensorBoard
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import product
from scipy.optimize import differential_evolution
#from dtw import dtw,accelerated_dtw
from scipy import signal
from pyts.decomposition import SingularSpectrumAnalysis


tf.keras.utils.get_custom_objects().clear() #clears serializables to rerun custom layers

#DEFINE CUSTOM LAYER CLASS
class MyCustomLayer(Layer):
    
    #INIT METHOD
    def __init__(self, output_dim,**kwargs):
        self.output_dim = output_dim
        super(MyCustomLayer,self).__init__(**kwargs)
    
    #GET CONFIG METHOD
    def get_config(self):
        config = super(MyCustomLayer,self).get_config()
        config.update({"output_dim": self.output_dim})
        return config
        
    #BUILD METHOD
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',shape=(input_shape[1],self.output_dim),initializer='normal',trainable = True)
        super(MyCustomLayer,self).build(input_shape)
    
    #CALL METHOD FOR TRAINING
    def call(self,input_data):
        return K.dot(input_data, self.kernel)
    
    #COMPUTE OUTPUT SHAPE METHOD
    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.output_dim)

#DEFINE CUSTOM DENSE LAYER
#@tf.keras.utils.register_keras_serializable()
class CustomDense(Layer):
    #INIT METHOD
    def __init__(self,units=32,activation='None',trainable=True):
        super(CustomDense, self).__init__() #inherit layer arguements
        self.units = units #neurons/computation units
        self.trainable=trainable
      
        if activation != 'None':
            self.activation = tf.keras.activations.get(activation)
        else:
            self.activation=activation
    
    #BUILD METHOD
    def build(self, input_shape): 
     
        # input shape 
        shape_w = input_shape[-1]
    
        # weights are created as per the no. of neurons // kernel = add.weight for custom weight variable
        self.kernel = self.add_weight("kernel",shape=(int(shape_w),self.units),initializer='normal',trainable = True)
    
        # bias values are created as per the no. of neurons // bias = add.weight for custom bias vector
        self.bias = self.add_weight(name='bias',shape=(self.units),initializer='zeros',trainable=True)
        
        #Build model with specified variables and inherited ones given the input shape
        super(CustomDense,self).build(input_shape)
        
    #CALL METHOD FOR CALCULATIONS   
    def call(self, inputs):  
      
        if self.activation !='None':      
            tf.matmul(inputs, self.kernel)
            return self.activation(tf.matmul(inputs, self.kernel) + self.bias)
        else:
            return tf.matmul(inputs, self.kernel) + self.bias
        
        
    #GET_CONFIG TO LOG THE INIT PARAMETERS    
    def get_config(self):
        return {"units": self.units}

    def compute_output_shape(self, input_shape): 
        return (input_shape[0], self.units)


#DEFINE CUSTOM ATTENTION LAYER (SERIALIZABLE TO SAVE AND LOAD)
@tf.keras.utils.register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    def __init__(self, units, name=None, **kwargs):
        super(Attention, self).__init__(name=name)
        #super(Attention, self).__init__()
        self.units=units
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        #super(Attention, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute attention scores
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # Apply attention weights to input
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "W": self.W,
            "V": self.V,
            "units": self.units
        })
        return config

def get_models():
    models = list()
    models.append(LinearRegression())
    models.append(DecisionTreeRegressor())
    models.append(KNeighborsRegressor(n_neighbors=4))
    models.append(MLPRegressor(random_state=1, max_iter=4000, early_stopping=True)) #extra model to check MLP
    return models

def calculate_mape(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

#DENOISING FUNCTION
def fft_denoiser(x, n_components, to_real=True,name=''):
    """Fast fourier transform denoiser.
    
    Denoises data using the fast fourier transform.
    
    Parameters
    ----------
    x : numpy.array
        The data to denoise.
    n_components : int
        The value above which the coefficients will be kept.
    to_real : bool, optional, default: True
        Whether to remove the complex part (True) or not (False)
        
    Returns
    -------
    clean_data : numpy.array
        The denoised data.
        
    References
    ----------
    .. [1] Steve Brunton - Denoising Data with FFT[Python]
       https://www.youtube.com/watch?v=s2K1JfNR7Sc&ab_channel=SteveBrunton
    
    """
    n = len(x)
    
    # compute the fft
    fft = np.fft.fft(x, n)
    
    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = fft * np.conj(fft) / n
    
    #PLOT PSD IN FREQ (AD-HOC)
    #frq = (1/(0.001*n))*np.arange(n)
    #L=np.arange(1,np.floor(n/2),dtype='int')
    #pp.plot(frq[L],PSD[L])
    #pp.xlabel("Frequency (Hz)")
    #pp.ylabel("Power Spectral Density")
    #pp.legend()
    #pp.savefig(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\psd_charts\psd_'+name+'.jpg',dpi=300,bbox_inches="tight")
    #pp.show()
    
    print(PSD,PSD.shape)
    
    # keep high frequencies
    _mask = PSD > n_components
    fft = _mask * fft
    
    # inverse fourier transform
    clean_data = np.fft.ifft(fft)
    
    if to_real:
        clean_data = clean_data.real
    
    return clean_data


###SIMILARITY FUNCTIONS
def euclidean_distance(x,y):
  return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def manhattan_distance(x,y):
  return sum(abs(a-b) for a,b in zip(x,y))

def nth_root(value, n_root):
 root_value = 1/float(n_root)
 return round (Decimal(value) ** Decimal(root_value),3)
  
def minkowski_distance(x,y,p_value):
 return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)

def square_rooted(x):
   return round(sqrt(sum([a*a for a in x])),3)
  
def cosine_similarity(x,y):
 numerator = sum(a*b for a,b in zip(x,y))
 denominator = square_rooted(x)*square_rooted(y)
 return round(numerator/float(denominator),3)

###SIMILARITY METRICS FUNCTION
def similarity_metrics(x,y):
    eu_dis = euclidean_distance(x,y)
    man_dis = manhattan_distance(x,y)
    cos_dis = cosine_similarity(x,y)
    return eu_dis,man_dis,cos_dis


def model_evaluation(actual_inverted, predicted_inverted):
    mape_dnn = calculate_mape(actual_inverted, predicted_inverted)
    mse_dnn = mean_squared_error(actual_inverted, predicted_inverted)
    rmse_dnn = math.sqrt(mse_dnn)
    mae_dnn = mean_absolute_error(actual_inverted, predicted_inverted)

    #print(mape_dnn, mse_dnn, rmse_dnn, mae_dnn)
    return mape_dnn, mse_dnn, rmse_dnn, mae_dnn

def list_difference(list1,list2):
    temp3 = []
    for element in list1:
        if element not in list2:
            temp3.append(element)
 
    #print(temp3)
    return(temp3)

#CONVERT INPUT INTO 3-D ARRAY FOR LSTM
def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y

def lstm_network(X_train_lstm,X_test_lstm,y_train_lstm,timesteps,features,lstm_units,dense_units,output_units,passes):
    model = Sequential()
    model.add(InputLayer((timesteps,features)))
    model.add(LSTM(lstm_units))
    model.add(Dense(dense_units,'relu'))
    model.add(Dense(output_units,'linear'))
    model.compile(optimizer='adam', loss= 'mae')

    model.summary()
    model.fit(X_train_lstm, y_train_lstm.values, epochs = passes, verbose=1)
    yhat = model.predict(X_test_lstm)
    print(yhat,yhat.shape)
    return yhat

#H-SWISH LOSS FUNCTION IMPLEMENTATION
def hard_swish(x):
    return x*tf.nn.relu6(x+3)/6

def custom_MLP(neurons_h1,neurons_h2,neurons_out,input_dimension,name):
    
    #IN CASE OF CUSTOM ACTIVATION FUNCTION
    get_custom_objects().update({'hard_swish': Activation(hard_swish)})
    
    nn=Sequential()
    nn.add(Dense(neurons_h1,activation='relu',input_dim=input_dimension)) #first hidden layer feeding from input 
    if name=='customdense_layer':
        print("USING CUSTOM LAYER")
        #nn.add(MyCustomLayer(32, input_shape = (16,)))
        nn.add(CustomDense(neurons_h2, 'relu'))     #RELU OR HARD-SWISH
    else:
        nn.add(Dense(neurons_h2)) #second hidden layer
    nn.add(Dense(neurons_out)) #output
    
    #APPLYING TIME-BASED LEARNING RATE DECAY FOR SGD (OTHERWISE DEFAULT OPTIMIZER IS ADAM)
    epochs = 4000
    learning_rate = 0.0005
    decay_rate = 1e-6
    momentum = 0.8
    sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    
    nn.compile(loss='mae',optimizer=sgd)
    plot_model(nn, to_file='D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\dmlp_model.png')
    return nn

def DNN_run(trainset_X,trainset_Y,testset_X,testset_Y, neurons_h1, neurons_h2, neurons_out,input_dimension, bsize, eps, name):
    
    nn=custom_MLP(neurons_h1,neurons_h2,neurons_out,input_dimension,name)
    nn.summary()
    
    #EARLY STOPPING
    es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=100)
    bestmodelpath = r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\dnn_'+name+'.h5'
    mc = ModelCheckpoint(bestmodelpath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    
    history=nn.fit(trainset_X,trainset_Y,batch_size=bsize,validation_data=(testset_X,testset_Y),epochs=eps, verbose=2, callbacks=[es, mc])

    testPredict = nn.predict(testset_X)

    pp.plot(history.history['loss'], label='train')
    pp.plot(history.history['val_loss'], label='test')
    pp.legend()
    pp.savefig(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\dnn_loss_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    
    #RETURN TRAINING ERROR TERMS
    trainPredict = nn.predict(trainset_X)

    return testPredict, trainPredict

def baseline_LSTM(input_dim_1, input_dim_2, lstm_units,n_out):
    
    nn = Sequential()
    #First Layer
    nn.add(LSTM(lstm_units,input_shape=(input_dim_1, input_dim_2))) #input dim 1: timesteps, input dim 2: features
    #Output Layer
    nn.add(Dense(n_out))
    #Compile, choose loss and optimizer metrics
    nn.compile(loss='mae',optimizer='adam')
    
    plot_model(nn, to_file='D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\base_lstm_model.png') 
    return nn

def attention_lstm(input_dim_1, input_dim_2, lstm_units,n_out):
    inputs = tf.keras.Input(shape=(input_dim_1, input_dim_2))
    x = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = Attention(lstm_units,name='Attention')(x)
    x = tf.keras.layers.Dense(n_out)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    model.compile(loss='mae',optimizer='adam')
    plot_model(model, to_file='D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\attention_lstm_model.png')
    return model

def autoencoder_lstm(input_dim_1, input_dim_2, lstm_units,n_out):
    model = Sequential()
    model.add(LSTM(lstm_units, activation='relu', input_shape=(input_dim_1,input_dim_2), return_sequences=True))
    model.add(LSTM(lstm_units//2, activation='relu', return_sequences=True)) #True if there is another LSTM LAYER AFTER IT OTHERWISE FALSE
    model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(input_dim_1))
    model.add(LSTM(lstm_units//2, activation='relu', return_sequences=True))
    model.add(LSTM(lstm_units, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(input_dim_2)))
    
    model.compile(loss='mae',optimizer='adam')
    plot_model(model, to_file='D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\autoencoder_lstm_model.png')
    return model

def encoder_decoder_lstm(input_dim_1, input_dim_2, lstm_units,n_out):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(input_dim_1,input_dim_2))
    encoder = LSTM(lstm_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(1, n_out))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
    decoder_dense = Dense(n_out, activation='linear')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='mae')
    
    #Inference encoder model
    encoder_model = Model(encoder_inputs, encoder_states)
    
    #Inference decoder
    decoder_state_input_h = Input(shape=(lstm_units,))
    decoder_state_input_c = Input(shape=(lstm_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
    
    return model, encoder_model, decoder_model
    

def LSTM_run(trainset_X,trainset_Y,testset_X,testset_Y, lstm_units,n_out,eps,bsize, name):
    
    ##DEFINE LSTM INPUT SHAPES AND FEATURES (setting timesteps to t=24) TEMPORAL RESHAPING
    print(trainset_X.shape)
    
    timesteps = 24
    n_features = trainset_X.shape[1]
    """
    #TRAIN INPUT 3D PREPARATION
    X_train_lstm, y_train_lstm = temporalize(X = trainset_X, y = trainset_Y, lookback = timesteps)

    n_features = trainset_X.shape[1]
    X_train_lstm = np.array(X_train_lstm)
    y_train_lstm = np.array(y_train_lstm)
    X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], timesteps, n_features)
    #print(X_train_lstm,X_train_lstm.shape)
    #print(y_train_lstm,y_train_lstm.shape)
    
    #TEST INPUT 3D PREPARATION
    X_test_lstm, y_test_lstm = temporalize(X = testset_X, y = testset_Y, lookback = timesteps)

    n_features = testset_X.shape[1]
    X_test_lstm = np.array(X_test_lstm)
    y_test_lstm = np.array(y_test_lstm)
    X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], timesteps, n_features)
    #print(X_test_lstm,X_test_lstm.shape)
    #print(y_test_lstm,y_test_lstm.shape)
    
    ###
    """
    #REGULAR VANILLA RESHAPING (COULD USE THIS OR THE TEMPORAL ABOVE FOR EXPERIMENTS)
    #if 'autoencoder_lstm' in name:
    #    X_train_lstm = trainset_X.reshape((trainset_X.shape[0],1, trainset_X.shape[1]))
    #    X_test_lstm = testset_X.reshape((testset_X.shape[0],1, testset_X.shape[1]))
    
    #FOR MEM1 MODELS
    X_train_lstm = trainset_X.reshape((trainset_X.shape[0],1, trainset_X.shape[1]))
    X_test_lstm = testset_X.reshape((testset_X.shape[0],1, testset_X.shape[1]))
    
    y_train_lstm = trainset_Y
    y_test_lstm = testset_Y
    
    if 'attention_lstm' in name:
        print(name)
        nn=attention_lstm(X_train_lstm.shape[1], X_train_lstm.shape[2], lstm_units,n_out)
    elif 'autoencoder_lstm' in name:
        print(name)
        nn=autoencoder_lstm(X_train_lstm.shape[1], X_train_lstm.shape[2], lstm_units,n_out)
        nn.summary()
        
        #TRAINING OF AUTOENCODER AND SEND OUTPUT
        es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=100)
        bestmodelpath = r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\lstm_'+name+'.h5'
        mc = ModelCheckpoint(bestmodelpath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        #AUTO-TRAINING ON INPUT X
        history=nn.fit(X_train_lstm,X_train_lstm,batch_size=bsize,validation_data=(X_test_lstm,X_test_lstm),epochs=eps, verbose=2, callbacks=[es, mc])

        testPredict = nn.predict(X_test_lstm)

        pp.plot(history.history['loss'], label='train')
        pp.plot(history.history['val_loss'], label='test')
        pp.legend()
        pp.savefig(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\lstm_loss_'+name+'.jpg',dpi=300,bbox_inches="tight")
        pp.show()
        
        #RETURN TRAINING ERROR TERMS
        trainPredict = nn.predict(X_train_lstm)
        
        #TAKE ENCODER OUTPUT
        encoder = Model(inputs=nn.inputs, outputs=nn.layers[2].output)
        train_encoded = encoder.predict(X_train_lstm)
        validation_encoded = encoder.predict(X_test_lstm)
        print('Encoded time-series shape', train_encoded.shape)
        print('Encoded time-series sample', train_encoded[0])
        print('Encoded time-series shape', validation_encoded.shape)
        print('Encoded time-series sample', validation_encoded[0])
        
        np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\X_train_"+name+"_encoded.npy",train_encoded)
        np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\X_test_"+name+"_encoded.npy",validation_encoded)
        

        return testPredict, trainPredict, X_test_lstm, y_test_lstm, y_train_lstm
    
    elif name == 'encoder_decoder_lstm': #not currently performing as intended
        nn, infenc, infdec =encoder_decoder_lstm(X_train_lstm.shape[1], X_train_lstm.shape[2], lstm_units,n_out)
        nn.summary()
        
        #TRAINING OF AUTOENCODER AND SEND OUTPUT
        es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=100)
        bestmodelpath = r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\lstm_'+name+'.h5'
        mc = ModelCheckpoint(bestmodelpath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        #ENCODER-DECODER TRAINING
        decoder_input_train = y_train_lstm.reshape(y_train_lstm.shape[0], 1, y_train_lstm.shape[1])
        decoder_input_test = y_test_lstm.reshape(y_test_lstm.shape[0], 1, y_test_lstm.shape[1])
        history=nn.fit([X_train_lstm,decoder_input_train],y_train_lstm,batch_size=bsize,validation_data=([X_test_lstm,decoder_input_test],y_test_lstm),epochs=eps, verbose=2, callbacks=[es, mc])

        testPredict = nn.predict([X_test_lstm,decoder_input_test])

        pp.plot(history.history['loss'], label='train')
        pp.plot(history.history['val_loss'], label='test')
        pp.legend()
        pp.savefig(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\lstm_loss_'+name+'.jpg',dpi=300,bbox_inches="tight")
        pp.show()
        
        #RETURN TRAINING ERROR TERMS
        trainPredict = nn.predict([X_train_lstm,decoder_input_train])

        return testPredict, trainPredict, X_test_lstm, y_test_lstm, y_train_lstm
        
        
    else:
        print(name)
        nn=baseline_LSTM(X_train_lstm.shape[1], X_train_lstm.shape[2], lstm_units,n_out)
    nn.summary()
    
    #EARLY STOPPING
    es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=100)
    bestmodelpath = r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\lstm_'+name+'.h5'
    mc = ModelCheckpoint(bestmodelpath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    
    history=nn.fit(X_train_lstm,y_train_lstm,batch_size=bsize,validation_data=(X_test_lstm,y_test_lstm),epochs=eps, verbose=2, callbacks=[es, mc])

    testPredict = nn.predict(X_test_lstm)
    

    pp.plot(history.history['loss'], label='train')
    pp.plot(history.history['val_loss'], label='test')
    pp.legend()
    pp.savefig(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\lstm_loss_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    
    #RETURN TRAINING ERROR TERMS
    trainPredict = nn.predict(X_train_lstm)
    

    return testPredict, trainPredict, X_test_lstm, y_test_lstm, y_train_lstm


def eval_multiseq(y_test, y_pred_test, name):
    if y_test.ndim==1:
        print("1D OUTPUT")
        if type(y_test) == np.ndarray:
            mape, mse, rmse, mae = model_evaluation(y_test, y_pred_test)
            print(mape, mse, rmse, mae)
        else:
            mape, mse, rmse, mae = model_evaluation(y_test.values, y_pred_test)
            print(mape, mse, rmse, mae)
    else:
        print("SEQUENCE EVALUATION BEGINS: ")
        print(str(y_pred_test.shape[1]) + " OUTPUT SEQUENCES")
        print(type(y_test))
        metric_storage=[]
        for i in range(y_pred_test.shape[1]):
            if type(y_test) == np.ndarray:
                mape, mse, rmse, mae = model_evaluation(y_test[:,i], y_pred_test[:,i])
                print(mape, mse, rmse, mae)
                metric_object = [mape, mse, rmse, mae]
                metric_storage.append(metric_object)
            else:
                mape, mse, rmse, mae = model_evaluation(y_test.values[:,i], y_pred_test[:,i])
                print(mape, mse, rmse, mae)
                metric_object = [mape, mse, rmse, mae]
                metric_storage.append(metric_object)
        metric_np = np.array(metric_storage)
        np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\metrics_"+name+"_model.npy",metric_np)

def naive_lr(X_train_norm, y_train, X_test_norm):
    lr_model = LinearRegression()
    reg = lr_model.fit(X_train_norm, y_train)
    y_hat=reg.predict(X_test_norm)
    return y_hat

def feature_preparation(X_fnames,y_fnames,data):
    #SPLIT INTO TRAIN AND TEST SET
    X = data[X_fnames]
    y = data[y_fnames]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,shuffle = False)

    #NORMALIZE DATASET
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_norm, X_test_norm

def calculate_dtw(series1, series2):
    timeseries1=to_time_series(series1)
    timeseries2=to_time_series(series2)
    #print(timeseries1)

    #soft_score=soft_dtw(timeseries1, timeseries2, gamma=.1)
    score=dtw(timeseries1, timeseries2)
    #print(soft_score)
    return score

def find_average(lst): 
    return sum(lst) / len(lst)


#set option for display
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',25)

####MAIN BODY
#LOAD THE DATASET
hourly_res_dataset = "gr_dataset_100t.csv"
path = "D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\\"+hourly_res_dataset

d_main=pd.read_csv(path) # Load dataset 
#print(d_main)
#print(d_main.shape)
#print(d_main.info(verbose=True))


"""
#CORRELATION ANALYSIS WITH CORR>ABS(0.4,0.6,0.8,0.9)
#Data Correlations with target sequences
all_selected_features = []
target_list = []
for i in range(24):
    target_name = "GR_load_actual_entsoe_transparency"
    if i==0:
        lagname=""
    else:
        lagname="lag_"+str(i)
        
    target_name = target_name + lagname
    target_list.append(target_name)
    print(target_name)
    correlation_matrix = d_main.corr()
    abs_corr = correlation_matrix.loc[target_name].abs()
    corr_slice = abs_corr[abs_corr>0.9].dropna()
    print(abs_corr,abs_corr.shape)
    print(corr_slice, corr_slice.shape)
    selected_features = corr_slice.index.tolist()
    selected_features.remove(target_name) #Features selected from pearson correlation analysis
    print(selected_features)
    all_selected_features = all_selected_features + selected_features
    #print(all_selected_features)

all_selected_features = list(dict.fromkeys(all_selected_features)) #REMOVE DUPLICATES
all_selected_features = list_difference(all_selected_features, target_list) #EXCLUDE TARGET VARIABLES
print(all_selected_features,len(all_selected_features))
print("TARGETS")
print(target_list)

#SAVE CORRELATION ANALYSIS RESULTS TO CSV
feat_dict = {'selected_features': all_selected_features, 'targets': target_list}
feat_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in feat_dict.items() ])) #CREATE DATAFRAME FROM DICTIONARY OF UNEVEN ARRAYS
feat_df.to_csv(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\corr_features_09.csv',index=False)
#END OF CORRELATION ANALYSIS BLOCK
exit()    
"""

#ORIGINAL DATASET FEATURE NAME LISTS FOR HANDLING
#print(d_main.columns.tolist())
wind_features = d_main.columns[pd.Series(d_main.columns).str.startswith('GR_wind_onshore_generation')].tolist()
target_names = d_main.columns[0:24].tolist()
input_names = d_main.columns[24:d_main.shape[1]].tolist()
#print(target_names)
#print(input_names,len(input_names))
print(wind_features)

# PREPARE DATASET AFTER CORR ANALYSIS
# LOAD CORRELATION RESULTS (CORRELATION-BASED MODELING)
corr_df = pd.read_csv(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\corr_features.csv')
#print(corr_df['selected_features'].values) # X names
#print(corr_df['targets'].dropna().values) # Y names

#FEATURE CONFIG #0: FULL 300 FEATURES - 24 TARGETS| 276 INPUTS
#X_train, X_test, y_train, y_test, X_train_norm, X_test_norm = feature_preparation(input_names, target_names, d_main.copy())

#FEATURE CONFIG #1 (BEST): CORR-ABS> 0.4,0.6,0.8,0.9
X_train, X_test, y_train, y_test, X_train_norm, X_test_norm = feature_preparation(corr_df['selected_features'].values, corr_df['targets'].dropna().values, d_main.copy())

#FEATURE CONFIG #2 : INPUT HAS ONLY WIND FEATURES
#X_train, X_test, y_train, y_test, X_train_norm, X_test_norm = feature_preparation(wind_features, corr_df['targets'].dropna().values, d_main.copy())

#print(X_train, X_test, y_train, y_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("FINAL SHAPE")
print(X_train_norm.shape, X_test_norm.shape, y_train.shape, y_test.shape)

#BUILD NAIVE LR BASELINE MODEL PREDICTION
y_hat = naive_lr(X_train_norm, y_train, X_test_norm)


pp.plot(y_hat[:,0],label='Predicted') # [:,i] for ith sequence
pp.plot(y_test.values[:,0],label='Actual')
pp.legend()
pp.show()
print("NAIVE LR")
eval_multiseq(y_test, y_hat, name='lr_baseline_all0')
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_lr_baseline.npy",y_hat)

y_hat_train = naive_lr(X_train_norm, y_train, X_train_norm)
eval_multiseq(y_train, y_hat_train, name='lr_baseline_train_all0')
#exit()

#TRY AND INCLUDE WIND FEATURES SEE IF IT IMPROVES AND IF IT DOESNT TRY AND SEE IF WE CAN EXTRACT ANYTHING MEANINGFUL FROM THAT SEQ
#TO DO: RUN THE MLP WITH THE CORRELATED CONFIGS AND TRY TO CREATE A CUSTOM LAYER

"""
#GET BASELINE PREDICTIONS 
prediction_vectors = list()
peak_indices_vectors = list()
peak_values_vectors = list ()
nonpeak_indices_vectors = list()
nonpeak_values_vectors = list()
for model in get_models():
    print(model)
    model.fit(X_train_norm,y_train.values)
    y_hat = model.predict(X_test_norm)
    mape_model, mse_model, rmse_model, mae_model = model_evaluation(y_test.values[:,0],y_hat[:,0])
    print(mape_model, mse_model, rmse_model, mae_model)
    prediction_vectors.append(y_hat[:,0])

#END OF BASELINE PREDICTIONS
"""


#DEVELOP BASELINE MODELS WITH EARLY STOPPING [DNN, LSTM, ATTENTION LSTM]

"""
##TEST CUSTOM DENSE LAYER FUNCTIONALITY
my_lay=CustomDense(units=X_train_norm.shape[1])
layer=my_lay.build((X_train_norm.shape[0],X_train_norm.shape[1]))

a= tf.constant(X_train_norm.astype('float32'), shape=[X_train_norm.shape[0],X_train_norm.shape[1]])
print("Input Data::\n ",a)
print("Layer Output Value::\n",my_lay.call(a))
print("Layer Weights Value:::\n",my_lay.variables[0])

#Operations on weights and conversions
print(tf.convert_to_tensor(my_lay.variables[0].numpy(), np.float32))
#PRINT weights of 1st feature
print(my_lay.variables[0][0])

print("Layer bias Value:::\n",my_lay.variables[1])
print("Number of neurons::",my_lay.get_config())
print("Output Shape of layer: ",my_lay.compute_output_shape((1,1)))

exit()
#END OF CUSTOM DENSE LAYER FUNCTIONALITY TEST
"""

"""
##RUN EXPERIMENTS ON CUSTOM MLP AND SAVE THEM/EXPERIMENTS ON STRUCTURE
run_name='customdense_layer'
y_pred_test,y_pred_train = DNN_run(trainset_X=X_train_norm,trainset_Y=y_train.values,testset_X=X_test_norm,testset_Y=y_test.values, neurons_h1 = X_train_norm.shape[1], neurons_h2=88, neurons_out = y_train.shape[1],input_dimension =  X_train_norm.shape[1], bsize=72, eps=10, name=run_name)
print(y_pred_test,y_pred_test.shape)
print(y_pred_train,y_pred_train.shape)
#END OF MLP DNN EXPERIMENTS FUNCTION
"""

"""
##RUN EXPERIMENTS ON LSTM AND SAVE THEM ['base_lstm','attention_lstm','autoencoder_lstm','temporal_example','mem24','encoder_decoder_lstm']
run_name='encoder_decoder_lstm'
y_pred_test,y_pred_train, evalX, evalY = LSTM_run(trainset_X=X_train_norm,trainset_Y=y_train.values,testset_X=X_test_norm,testset_Y=y_test.values, lstm_units=100,n_out=24,eps=100,bsize=72, name=run_name)
print(y_pred_test,y_pred_test.shape)
print(y_pred_train,y_pred_train.shape)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\X_test_eval.npy",evalX)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_test_eval.npy",evalY)
#END OF LSTM EXPERIMENTS FUNCTION
"""

"""
#LOAD SAVED ATTENTION LSTM MODEL AND EVALUATE
evalX = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\X_test_eval.npy")
evalY = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_test_eval.npy")
attention_model = load_model(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\lstm_attention_lstm.h5')
y_pred_test=attention_model.predict(evalX)
print(y_pred_test,y_pred_test.shape)
eval_multiseq(evalY, y_pred_test)

attention = tf.keras.Model(inputs=attention_model.input, outputs=attention_model.get_layer("Attention").output)
att_scores=attention.predict(evalX)
print(att_scores,att_scores.shape)

"""

"""
#LOAD SAVED LSTM MODEL AND EVALUATE
evalX = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\X_test_eval.npy")
evalY = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_test_eval.npy")
base_lstm = load_model(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\lstm_mem24.h5')
y_pred_test=base_lstm.predict(evalX)
print(y_pred_test,y_pred_test.shape)
eval_multiseq(evalY, y_pred_test)

#GET LSTM WEIGHTS
units = int(int(base_lstm.layers[0].trainable_weights[0].shape[1])/4)
print("No units: ", units)

W = base_lstm.layers[0].get_weights()[0]
U = base_lstm.layers[0].get_weights()[1]
b = base_lstm.layers[0].get_weights()[2]

#KERNEL WEIGHTS for input, forget, cell state, output gates
print("KERNEL WEIGHTS")
W_i = W[:, :units]
W_f = W[:, units: units * 2]
W_c = W[:, units * 2: units * 3]
W_o = W[:, units * 3:]
print(W_i,W_i.shape,W_f,W_f.shape,W_c,W_c.shape,W_o,W_o.shape)
#RECURRENT WEIGHTS for input, forget, cell state, output gates
print("RECURRENT WEIGHTS")
U_i = U[:, :units]
U_f = U[:, units: units * 2]
U_c = U[:, units * 2: units * 3]
U_o = U[:, units * 3:]
print(U_i,U_i.shape,U_f,U_f.shape,U_c,U_c.shape,U_o,U_o.shape)
#BIAS for input, forget, cell state, output gates
print("BIASES")
b_i = b[:units]
b_f = b[units: units * 2]
b_c = b[units * 2: units * 3]
b_o = b[units * 3:]
print(b_i,b_i.shape,b_f,b_f.shape,b_c,b_c.shape,b_o,b_o.shape)
"""

##DEVELOP FORECASTING DATASET FURTHER
print("TRAINSET",X_train_norm,X_train_norm.shape)

#GET MATRIX PROFILES OF INPUT FEATURES
#print(X_train_norm[:,0],X_train_norm[:,0].shape)
window_size=24
#matrix_profile = stumpy.stump(X_train_norm[:,0], m=window_size)
#print(matrix_profile,matrix_profile.shape)
#GET ALL-CHAIN SET AND UNANCHORED CHAIN
#all_chain_set, unanchored_chain = stumpy.allc(matrix_profile[:, 2], matrix_profile[:, 3])
#print(unanchored_chain,unanchored_chain.shape)
#for i in range(unanchored_chain.shape[0]):
#    y = X_train_norm[unanchored_chain[i]:unanchored_chain[i]+window_size,0]
#    pp.plot(y)
#    pp.show()

#matrix_profile = stumpy.stump(y_train.values[:,0], m=window_size)
#print(matrix_profile,matrix_profile.shape)
#GET ALL-CHAIN SET AND UNANCHORED CHAIN
#all_chain_set, unanchored_chain = stumpy.allc(matrix_profile[:, 2], matrix_profile[:, 3])
#print(unanchored_chain,unanchored_chain.shape)
#for i in range(unanchored_chain.shape[0]):
#    y = y_train.values[unanchored_chain[i]:unanchored_chain[i]+window_size,0]
#    pp.plot(y)
#    pp.show()

"""
#SSA DECOMPOSITION OF XTRAIN
print("SSA DECOMPOSITION")
#GET SSA COMPONENTS OF TRAIN INPUT FEATURES (CREATE/SAVE TRAIN_SSA_TREND, TRAIN_SSA_SEASONAL, TRAIN_SSA_RESIDUAL NUMPY ARRAYS OF SHAPE [40179,176] EACH)
train_ssa_trend=np.zeros(shape=(X_train_norm.shape[0],X_train_norm.shape[1]))
train_ssa_seasonal=np.zeros(shape=(X_train_norm.shape[0],X_train_norm.shape[1]))
train_ssa_residual=np.zeros(shape=(X_train_norm.shape[0],X_train_norm.shape[1]))

ssa = SingularSpectrumAnalysis(window_size=window_size, groups="auto")
for k in range(X_train_norm.shape[1]):
    X_train_ssa = ssa.fit_transform(X_train_norm[:,k].reshape(1, -1)) #GET SSA OF k-th input train feature
    print(X_train_ssa,X_train_ssa.shape)
    for i in range(X_train_ssa.shape[1]): #i=0: trend, i=1: seasonal, i=2: residual
        print(X_train_ssa[0,i,:],X_train_ssa[0,i,:].shape)
        ssa_component = X_train_ssa[0,i,:]
        if i==0:
            train_ssa_trend[:,k]=ssa_component
        elif i==1:
            train_ssa_seasonal[:,k]=ssa_component
        else:
            train_ssa_residual[:,k]=ssa_component
        #pp.plot(X_train_ssa[0,i,:])
        #pp.show()

print("TRAIN SSA DATASETS")
print(train_ssa_trend,train_ssa_trend.shape)
print(train_ssa_seasonal,train_ssa_seasonal.shape)
print(train_ssa_residual,train_ssa_residual.shape)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_ssa_trend.npy",train_ssa_trend)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_ssa_seasonal.npy",train_ssa_seasonal)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_ssa_residual.npy",train_ssa_residual)


#GET SSA COMPONENTS OF TEST INPUT FEATURES (CREATE/SAVE TEST_SSA_TREND, TEST_SSA_SEASONAL, TEST_SSA_RESIDUAL NUMPY ARRAYS OF SHAPE [X_test_norm.shape[0],X_test_norm.shape[1]] EACH)
test_ssa_trend=np.zeros(shape=(X_test_norm.shape[0],X_test_norm.shape[1]))
test_ssa_seasonal=np.zeros(shape=(X_test_norm.shape[0],X_test_norm.shape[1]))
test_ssa_residual=np.zeros(shape=(X_test_norm.shape[0],X_test_norm.shape[1]))
ssa = SingularSpectrumAnalysis(window_size=window_size, groups="auto")
for k in range(X_test_norm.shape[1]):
    X_test_ssa = ssa.fit_transform(X_test_norm[:,k].reshape(1, -1)) #GET SSA OF k-th input train feature
    print(X_test_ssa,X_test_ssa.shape)
    for i in range(X_test_ssa.shape[1]): #i=0: trend, i=1: seasonal, i=2: residual
        print(X_test_ssa[0,i,:],X_test_ssa[0,i,:].shape)
        ssa_component = X_test_ssa[0,i,:]
        if i==0:
            test_ssa_trend[:,k]=ssa_component
        elif i==1:
            test_ssa_seasonal[:,k]=ssa_component
        else:
            test_ssa_residual[:,k]=ssa_component
        #pp.plot(X_train_ssa[0,i,:])
        #pp.show()

print("TEST SSA DATASETS")
print(test_ssa_trend,test_ssa_trend.shape)
print(test_ssa_seasonal,test_ssa_seasonal.shape)
print(test_ssa_residual,test_ssa_residual.shape)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_ssa_trend.npy",test_ssa_trend)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_ssa_seasonal.npy",test_ssa_seasonal)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_ssa_residual.npy",test_ssa_residual)

exit() #END OF SSA DECOMPOSITION ON TRAIN AND TEST INPUTS 
"""
"""
print("LOESS DECOMPOSITION OF XTRAIN")
train_loess_trend=np.zeros(shape=(X_train_norm.shape[0],X_train_norm.shape[1]))
train_loess_seasonal=np.zeros(shape=(X_train_norm.shape[0],X_train_norm.shape[1]))
train_loess_residual=np.zeros(shape=(X_train_norm.shape[0],X_train_norm.shape[1]))

for k in range(X_train_norm.shape[1]):
    print("Examining feature "+str(k))
    stl = STL(X_train_norm[:,k], period=24,robust=True)
    stl_res = stl.fit()
    train_loess_trend[:,k]=stl_res.trend
    train_loess_seasonal[:,k]=stl_res.seasonal
    train_loess_residual[:,k]=stl_res.resid
    
    #fig = stl_res.plot()

print("TRAIN LOESS DATASETS")
print(train_loess_trend,train_loess_trend.shape)
print(train_loess_seasonal,train_loess_seasonal.shape)
print(train_loess_residual,train_loess_residual.shape)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_loess_trend.npy",train_loess_trend)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_loess_seasonal.npy",train_loess_seasonal)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_loess_residual.npy",train_loess_residual)

print("LOESS DECOMPOSITION OF XTEST")
test_loess_trend=np.zeros(shape=(X_test_norm.shape[0],X_test_norm.shape[1]))
test_loess_seasonal=np.zeros(shape=(X_test_norm.shape[0],X_test_norm.shape[1]))
test_loess_residual=np.zeros(shape=(X_test_norm.shape[0],X_test_norm.shape[1]))

for k in range(X_test_norm.shape[1]):
    print("Examining feature "+str(k))
    stl = STL(X_test_norm[:,k], period=24,robust=True)
    stl_res = stl.fit()
    test_loess_trend[:,k]=stl_res.trend
    test_loess_seasonal[:,k]=stl_res.seasonal
    test_loess_residual[:,k]=stl_res.resid
    
    #fig = stl_res.plot()

print("TEST LOESS DATASETS")
print(test_loess_trend,test_loess_trend.shape)
print(test_loess_seasonal,test_loess_seasonal.shape)
print(test_loess_residual,test_loess_residual.shape)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_loess_trend.npy",test_loess_trend)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_loess_seasonal.npy",test_loess_seasonal)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_loess_residual.npy",test_loess_residual)

exit() #END OF LOESS DECOMPOSITION
"""
"""
print("EMD ANALYSIS ON TRAIN FEATURES")
imf_train = emd.sift.sift(X_train_norm[:,0],max_imfs=5)
print(imf_train.shape) #INTRINSIC MODE FUNCTIONS FOR A FEATURE
print(imf_train)
for k in range(1,X_train_norm.shape[1]):
    imf_next = emd.sift.sift(X_train_norm[:,k],max_imfs=5)
    
    imf_train = np.concatenate((imf_train,imf_next),axis=1)
    print(imf_train.shape)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_emd.npy",imf_train)

print("EMD ANALYSIS ON TEST FEATURES")
imf_test = emd.sift.sift(X_test_norm[:,0],max_imfs=5)
print(imf_test.shape) #INTRINSIC MODE FUNCTIONS FOR A FEATURE
print(imf_test)
for k in range(1,X_test_norm.shape[1]):
    imf_next = emd.sift.sift(X_test_norm[:,k],max_imfs=5)
    
    imf_test = np.concatenate((imf_test,imf_next),axis=1)
    print(imf_test.shape)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_emd.npy",imf_test)

#IP, IF, IA = emd.spectra.frequency_transform(imf, 24, 'hilbert')
    #freq_range = (0.1, 10, 80, 'log')
    #f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)
    #emd.plotting.plot_imfs(imf)

#END OF EMD DECOMPOSITION
"""


##FORECASTING MODELS BASED ON COMPONENTS OF INPUT
#STEP 1: LOAD COMPONENT ARRAYS
#TRAIN SETS [SSA,LOESS,EMD]    
X_train_ssa_trend = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_ssa_trend.npy")
X_train_ssa_seasonal = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_ssa_seasonal.npy")
X_train_ssa_residual = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_ssa_residual.npy")

X_train_loess_trend = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_loess_trend.npy")
X_train_loess_seasonal = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_loess_seasonal.npy")
X_train_loess_residual = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_loess_residual.npy")

X_train_emd = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\train_emd.npy")

#TEST SETS [SSA,LOESS,EMD]    
X_test_ssa_trend = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_ssa_trend.npy")
X_test_ssa_seasonal = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_ssa_seasonal.npy")
X_test_ssa_residual = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_ssa_residual.npy")

X_test_loess_trend = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_loess_trend.npy")
X_test_loess_seasonal = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_loess_seasonal.npy")
X_test_loess_residual = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_loess_residual.npy")

X_test_emd = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\test_emd.npy")

component_datasets_train = [X_train_ssa_trend,X_train_ssa_seasonal,X_train_ssa_residual,X_train_loess_trend,X_train_loess_seasonal,X_train_loess_residual,X_train_emd]
names_datasets_train = ["X_train_ssa_trend","X_train_ssa_seasonal","X_train_ssa_residual","X_train_loess_trend","X_train_loess_seasonal","X_train_loess_residual","X_train_emd"]

component_datasets_test = [X_test_ssa_trend,X_test_ssa_seasonal,X_test_ssa_residual,X_test_loess_trend,X_test_loess_seasonal,X_test_loess_residual,X_test_emd]


"""
#STEP 2: GET FEATURE IMPORTANCES FROM ALL COMPONENTS
#DATASETS OF ALL COMPONENTS
X_train_components = np.concatenate((X_train_ssa_trend,X_train_ssa_seasonal,X_train_ssa_residual,X_train_loess_trend,X_train_loess_seasonal,X_train_loess_residual,X_train_emd),axis=1)
X_test_components = np.concatenate((X_test_ssa_trend,X_test_ssa_seasonal,X_test_ssa_residual,X_test_loess_trend,X_test_loess_seasonal,X_test_loess_residual,X_test_emd),axis=1)
print(X_train_components,X_train_components.shape)
print(X_test_components,X_test_components.shape)

for n,c in enumerate(component_datasets_train):
    # define the model
    model = XGBRegressor()
    # fit the model
    model.fit(c, y_train.values)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    np_imp=np.zeros(shape=(c.shape[1],2))
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
        np_imp[i,:]=[i,v]

    print("IMPORTANCE ARRAY OF "+names_datasets_train[n],np_imp)
    np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\importance_"+names_datasets_train[n]+".npy",np_imp)

print("END OF IMPORTANCE COLLECTION")
"""


#STEP 3: FIND MOST IMPORTANT COMPONENTS
for n,v in enumerate(names_datasets_train):
    c_loaded = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\decomposition\\importance_"+names_datasets_train[n]+".npy")
    c_annotated = np.concatenate((c_loaded, n*np.ones(shape=(c_loaded.shape[0],1))),axis=1)
    print(c_annotated,c_annotated.shape)
    if n==0:
        c_all = c_annotated
    else:
        c_all = np.concatenate((c_all, c_annotated),axis=0)

print(c_all,c_all.shape)
#SORT BY IMPORTANCE SCORES
c_all_sorted=c_all[c_all[:, 1].argsort()[::-1]]
print(c_all_sorted)

print(c_all_sorted[200,:])

#CREATE XTRAIN AND XTEST DATASET WITH 200 MOST IMPORTANT COMPONENTS
for k in range(200):
    comp_i=int(c_all_sorted[k,2])
    feat_col=int(c_all_sorted[k,0])
    print(comp_i,feat_col)
    #fetch features
    #Train
    selected_component_train=component_datasets_train[comp_i]
    selected_column_train = selected_component_train[:,feat_col]
    #Test
    selected_component_test=component_datasets_test[comp_i]
    selected_column_test = selected_component_test[:,feat_col]
    if k==0: 
        X_train_important = selected_column_train[:,None]
        X_test_important = selected_column_test[:,None]
    else:
        X_train_important = np.concatenate((X_train_important,selected_column_train[:,None]),axis=1)
        X_test_important = np.concatenate((X_test_important,selected_column_test[:,None]),axis=1)

print(X_train_important,X_train_important.shape)
print(X_test_important,X_test_important.shape)

#STEP 4: TRAIN REGRESSORS AND EVALUATE
#print(X_test_loess_trend)
#exit()
# KERNEL MODEL #1: NAIVE LR
y_hat = naive_lr(X_train_important, y_train, X_test_important)

pp.plot(y_hat[:,0],label='Predicted') # [:,i] for ith sequence
pp.plot(y_test.values[:,0],label='Actual')
pp.legend()
pp.show()
print("LR LOESS ALL")
print(y_hat)
print(y_test)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_lr_important.npy",y_hat)
#eval_multiseq(y_test, y_hat, name='lr_important_components')

y_hat_train = naive_lr(X_train_important, y_train, X_train_important)
#eval_multiseq(y_train, y_hat_train, name='lr_important_components_train')
#exit()

# KERNEL MODEL #2: DNN (BEST)
#run_name='dnn_important_components'
#y_pred_test,y_pred_train = DNN_run(trainset_X=X_train_important,trainset_Y=y_train.values,testset_X=X_test_important,testset_Y=y_test.values, neurons_h1 = X_train_important.shape[1], neurons_h2=X_train_important.shape[1]//2, neurons_out = y_train.shape[1],input_dimension =  X_train_important.shape[1], bsize=72, eps=4000, name=run_name)
#print(y_pred_test,y_pred_test.shape)
#print(y_pred_train,y_pred_train.shape)
#eval_multiseq(y_test, y_pred_test,name = 'dnn_important_components')

#LOAD MODEL
#imp_dnn = load_model(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\dnn_dnn_important_components.h5')
#y_pred_test=imp_dnn.predict(X_test_important)
#y_pred_train=imp_dnn.predict(X_train_important)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_dnn_important.npy",y_pred_test)

#EVALUATE TRAINING PERFORMANCE
#print("TRAINING PERFORMANCE")
#eval_multiseq(y_train.values, y_pred_train, name = 'dnn_important_components_train')

#KERNEL MODEL #3: LSTM-MEM24 AND MEM1 [lstm_important_components, lstm_mem1_important_components]
#run_name='lstm_mem1_important_components'
#y_pred_test,y_pred_train, evalX, evalY_test, evalY_train = LSTM_run(trainset_X=X_train_important,trainset_Y=y_train.values,testset_X=X_test_important,testset_Y=y_test.values, lstm_units=X_train_important.shape[1],n_out=24,eps=4000,bsize=72, name=run_name)
#print(y_pred_test,y_pred_test.shape)
#print(y_pred_train,y_pred_train.shape)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_lstm_important_mem1.npy",y_pred_test)

#EVALUATE TRAINING PERFORMANCE
#print("TRAINING PERFORMANCE")
#eval_multiseq(evalY_train, y_pred_train, name = run_name+'_train')
#EVALUATE TESTING PERFORMANCE
#print("TESTING PERFORMANCE")
#eval_multiseq(evalY_test, y_pred_test, name = run_name+'_test')

#KERNEL MODEL #4: ATTENTION-LSTM WITH IMPORTANT COMPONENTS
#run_name='attention_lstm_mem1_important_components'
#y_pred_test,y_pred_train, evalX, evalY_test, evalY_train = LSTM_run(trainset_X=X_train_important,trainset_Y=y_train.values,testset_X=X_test_important,testset_Y=y_test.values, lstm_units=X_train_important.shape[1],n_out=24,eps=4000,bsize=72, name=run_name)
#print(y_pred_test,y_pred_test.shape)
#print(y_pred_train,y_pred_train.shape)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_attention_lstm_important_mem1.npy",y_pred_test)

#EVALUATE TRAINING PERFORMANCE
#print("TRAINING PERFORMANCE")
#eval_multiseq(evalY_train, y_pred_train, name = run_name+'_train')
#EVALUATE TESTING PERFORMANCE
#print("TESTING PERFORMANCE")
#eval_multiseq(evalY_test, y_pred_test, name = run_name+'_test')

#KERNEL MODEL #5: XGBREGRESSOR

#FOR LOSS FUNCTION INSPECTION
#evals_result = {}
#eval_s = [(X_train_important,y_train.values), (X_test_important, y_test.values)]

#gbm = XGBRegressor()
#gbm.fit(X_train_important,y_train.values,eval_metric=["mae"],eval_set=eval_s)

#results = gbm.evals_result()
#pp.plot(results['validation_0']['mae'], label='train')
#pp.plot(results['validation_1']['mae'], label='test')
#pp.xlabel("Boosting Rounds")
#pp.ylabel("Mean Absolute Error")
# show the legend
#pp.legend()
# show the plot
#pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\xgb_important_learning_curves.png', dpi=1000)
#pp.show()

#exit()
#END OF LOSS FUNTION INSPECTION

#xgb_important = XGBRegressor()
#xgb_important.fit(X_train_important,y_train.values)
#y_pred_test = xgb_important.predict(X_test_important)
#eval_multiseq(y_test, y_pred_test,name = 'xgb_important_components')
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_xgb_important.npy",y_pred_test)
#EVALUATE TRAINING PERFORMANCE
#y_pred_train = xgb_important.predict(X_train_important)
#eval_multiseq(y_train, y_pred_train,name = 'xgb_important_components_train')

#///COMPLEMENTARY BASE MODEL EXPERIMENTS ON DEFAULT DATASET
##EXTRA BASE 1: ATTENTION LSTM
#run_name='attention_lstm_baseline_mem1'
#y_pred_test,y_pred_train, evalX, evalY_test, evalY_train = LSTM_run(trainset_X=X_train_norm,trainset_Y=y_train.values,testset_X=X_test_norm,testset_Y=y_test.values, lstm_units=X_train_norm.shape[1],n_out=24,eps=4000,bsize=72, name=run_name)
#print(y_pred_test,y_pred_test.shape)
#print(y_pred_train,y_pred_train.shape)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_attention_lstm_baseline_mem1.npy",y_pred_test)

#EVALUATE TRAINING PERFORMANCE
#print("TRAINING PERFORMANCE")
#eval_multiseq(evalY_train, y_pred_train, name = run_name+'_train')
#EVALUATE TESTING PERFORMANCE
#print("TESTING PERFORMANCE")
#eval_multiseq(evalY_test, y_pred_test, name = run_name+'_test')

##EXTRA BASE 2: XGB BASE

#FOR LOSS FUNCTION INSPECTION
#evals_result = {}
#eval_s = [(X_train_norm, y_train.values), (X_test_norm, y_test.values)]

#gbm = XGBRegressor()
#gbm.fit(X_train_norm, y_train.values,eval_metric=["mae"],eval_set=eval_s)

#results = gbm.evals_result()
#pp.plot(results['validation_0']['mae'], label='train')
#pp.plot(results['validation_1']['mae'], label='test')
#pp.xlabel("Boosting Rounds")
#pp.ylabel("Mean Absolute Error")
# show the legend
#pp.legend()
# show the plot
#pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\xgb_base_learning_curves.png', dpi=1000)
#pp.show()

#exit()
#END OF LOSS FUNTION INSPECTION

#xgb_base = XGBRegressor()
#xgb_base.fit(X_train_norm,y_train.values)

#y_pred_test = xgb_base.predict(X_test_norm)
#pp.plot(y_test.values[:,0])
#pp.plot(y_pred_test[:,0])
#pp.show()
#exit()
#eval_multiseq(y_test, y_pred_test,name = 'xgb_baseline')
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_xgb_baseline.npy",y_pred_test)

#EVALUATE TRAINING PERFORMANCE
#y_pred_train = xgb_base.predict(X_train_norm)
#eval_multiseq(y_train, y_pred_train,name = 'xgb_baseline_train')

##EXTRA BASE 3: LSTM BASE [lstm_baseline, lstm_baseline_mem1]
#run_name='lstm_baseline_mem1'
#y_pred_test,y_pred_train, evalX, evalY_test, evalY_train = LSTM_run(trainset_X=X_train_norm,trainset_Y=y_train.values,testset_X=X_test_norm,testset_Y=y_test.values, lstm_units=X_train_norm.shape[1],n_out=24,eps=4000,bsize=72, name=run_name)
#print(y_pred_test,y_pred_test.shape)
#print(y_pred_train,y_pred_train.shape)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_lstm_baseline_mem1.npy",y_pred_test)

#EVALUATE TRAINING PERFORMANCE
#print("TRAINING PERFORMANCE")
#eval_multiseq(evalY_train, y_pred_train, name = run_name+'_train')
#EVALUATE TESTING PERFORMANCE
#print("TESTING PERFORMANCE")
#eval_multiseq(evalY_test, y_pred_test, name = run_name+'_test')


#STEP 5: INTERPRET AND PROCESS RESULTS
###METRICS ON TEST SET
##HYBRID COMPONENT VARIANTS (5)
dnn_important_metrics_test = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_dnn_important_components_model.npy")
attention_lstm_important_metrics_test = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_attention_lstm_mem1_important_components_test_model.npy")
lstm_important_metrics_test = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_lstm_mem1_important_components_test_model.npy")
xgb_important_metrics_test = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_xgb_important_components_model.npy")
lr_important_metrics_test = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_lr_important_components_model.npy")

#ENCAPSULATING HYBRID COMP METRICS
comp_test_list_all = [dnn_important_metrics_test,attention_lstm_important_metrics_test,lstm_important_metrics_test,xgb_important_metrics_test,lr_important_metrics_test]
comp_test_list_mape = [dnn_important_metrics_test[:,0],attention_lstm_important_metrics_test[:,0],lstm_important_metrics_test[:,0],xgb_important_metrics_test[:,0],lr_important_metrics_test[:,0]]

##BASELINE BENCHMARK MODELS (5)
dnn_base_test = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_dnn_baseline_model.npy")
attention_lstm_base_test = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_attention_lstm_baseline_mem1_test_model.npy")
lstm_base_test = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_lstm_baseline_mem1_test_model.npy")
xgb_base_test = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_xgb_baseline_model.npy")
lr_base_test = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_lr_baseline_model.npy")

#ENCAPSULATING BASE MODEL METRICS
base_test_list_all = [dnn_base_test,attention_lstm_base_test,lstm_base_test,xgb_base_test,lr_base_test]
base_test_list_mape = [dnn_base_test[:,0],attention_lstm_base_test[:,0],lstm_base_test[:,0],xgb_base_test[:,0],lr_base_test[:,0]]

###METRICS ON TRAIN SET
##HYBRID COMP VARIANTS (5)
dnn_important_metrics_train = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_dnn_important_components_train_model.npy")
attention_lstm_important_metrics_train = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_attention_lstm_mem1_important_components_train_model.npy")
lstm_important_metrics_train = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_lstm_mem1_important_components_train_model.npy")
xgb_important_metrics_train = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_xgb_important_components_train_model.npy")
lr_important_metrics_train = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_lr_important_components_train_model.npy")

#ENCAPSULATING TRAINING PERFORMANCE OF IMPORTANT COMPONENT MODELS
comp_train_list_all = [dnn_important_metrics_train,attention_lstm_important_metrics_train,lstm_important_metrics_train,xgb_important_metrics_train,lr_important_metrics_train]
comp_train_list_mape = [dnn_important_metrics_train[:,0],attention_lstm_important_metrics_train[:,0],lstm_important_metrics_train[:,0],xgb_important_metrics_train[:,0],lr_important_metrics_train[:,0]]

##BASELINE BENCHMARK MODELS (5)
dnn_base_train = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_dnn_baseline_train_model.npy")
attention_lstm_base_train = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_attention_lstm_baseline_mem1_train_model.npy")
lstm_base_train = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_lstm_baseline_mem1_train_model.npy")
xgb_base_train = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_xgb_baseline_train_model.npy")
lr_base_train = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\\metrics_lr_baseline_train_model.npy")

#ENCAPSULATING TRAINING PERFORMANCE OF BASE MODELS
base_train_list_all = [dnn_base_train,attention_lstm_base_train,lstm_base_train,xgb_base_train,lr_base_train]
base_train_list_mape = [dnn_base_train[:,0],attention_lstm_base_train[:,0],lstm_base_train[:,0],xgb_base_train[:,0],lr_base_train[:,0]]

#DEVELOP OUTPUT SEQUENCE SELECTOR BASED ON TRAINING PERFORMANCE
print("COMPONENT LIST TEST")
print(comp_test_list_all)
print(comp_test_list_mape[0].shape)

#DF ON COMP TEST FOR PROOFING
df_col_list = ['mape_dnn','mape_attlstm','mape_lstm','mape_xgb','mape_lr']
df_comp_test_mape = pd.DataFrame() 
for ind,col in enumerate(df_col_list):
    df_comp_test_mape[col]= comp_test_list_mape[ind]

df_comp_test_mape['min_seq_mape'] = df_comp_test_mape.idxmin(axis=1)

print("COMP TEST DATAFRAME")
print(df_comp_test_mape,df_comp_test_mape.shape)

#DF ON BASE TEST TO SEE THE DIFFERENCES
df_base_test_mape = pd.DataFrame() 
for ind,col in enumerate(df_col_list):
    df_base_test_mape[col]= base_test_list_mape[ind]

df_base_test_mape['min_seq_mape'] = df_base_test_mape.idxmin(axis=1)

print("BASE TEST DATAFRAME")
print(df_base_test_mape,df_base_test_mape.shape)

#DF ON COMP TRAIN FOR DISCOVERY
#df_col_list = ['mape_dnn','mape_attlstm','mape_lstm','mape_xgb','mape_lr']
#df_comp_train_mape = pd.DataFrame() 
#for ind,col in enumerate(df_col_list):
#    df_comp_train_mape[col]= comp_train_list_mape[ind]

#df_comp_train_mape['min_seq_mape'] = df_comp_train_mape.idxmin(axis=1)

#print("COMP TRAIN DATAFRAME")
#print(df_comp_train_mape,df_comp_train_mape.shape)

"""
#CREATE TABLES AND GRAPHS
#TABLE 1: AVERAGE METRICS FOR COMP AND BASE MODELS AND PER-SEQ GRAPHS
#i) BASE MODELS
X_axis = np.arange(base_test_list_all[0].shape[0])
labels = np.arange(base_test_list_all[0].shape[0])
name_list = ['DNN','Att-LSTM','LSTM','XGB','LR']
name_list_comp = ['CC-DNN','CC-Att-LSTM','CC-LSTM','CC-XGB','CC-LR'] #Combinatorial Component (CC)
for ind,base_item in enumerate(base_test_list_all):
    average_metrics = np.average(base_item,axis=0)
    print(name_list[ind])
    print(average_metrics.tolist()) #AVG METRICS FOR BASE
    
    metric_list = ['MAPE','MSE','RMSE','MAE']
    metric_list_full = ['Mean Absolute Percentage Error','Mean Squared Error','Root Mean Squared Error','Mean Absolute Error']
    
    for m_i,metric in enumerate(metric_list):
        m_df=pd.DataFrame()
        m_df[name_list[ind]]=np.flip(base_item[:,m_i])
        m_df[name_list_comp[ind]] = np.flip(comp_test_list_all[ind][:,m_i])
        m_df.plot.bar()
        pp.xticks(rotation=0)
        pp.xlabel("Predicted Sequence")
        pp.ylabel(metric_list_full[m_i])
        #pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\'+metric_list[m_i]+'_'+name_list[ind]+'.png', dpi=1000)
        pp.show()
        
        #pp.bar(X_axis,np.flip(base_item[:,m_i]),label=name_list[ind]) #BASE MODELS
        #pp.bar(X_axis,np.flip(comp_test_list_all[ind][:,m_i]),label=name_list_comp[ind]) #CC MODELS
        #pp.xticks(X_axis)
        #pp.xlabel("Predicted Sequence")
        #pp.ylabel(metric_list_full[m_i])
        #pp.title("Feature Representativeness")
        #pp.legend()
        #pp.show()
    
#ii) COMP MODELS
name_list = ['dnn_comp','attlstm_comp','lstm_comp','xgb_comp','lr_comp']
for ind,comp_item in enumerate(comp_test_list_all):
    average_metrics = np.average(comp_item,axis=0)
    print(name_list[ind])
    print(average_metrics.tolist()) #AVG METRICS FOR COMP

exit()
#END OF TABLE-RELATED DATA
"""


#GET Y_TEST AND Y_PRED OF MODELS, FIND SHAPELETS IN Y_TEST AND CALCULATE WEIGHTED AVERAGE CHAIN CLOSENESS (weight function f(x)=(1/window_size)*x)
model_names=['DNN','ATT-LSTM','LSTM','XGB','LR']
#Y OF CC-MODELS
y_dnn_important = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_dnn_important.npy")
y_attention_lstm_important_mem1 = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_attention_lstm_important_mem1.npy")
y_lstm_important_mem1 = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_lstm_important_mem1.npy")
y_xgb_important = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_xgb_important.npy")
y_lr_important = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_lr_important.npy")

y_cc = [y_dnn_important,y_attention_lstm_important_mem1,y_lstm_important_mem1,y_xgb_important,y_lr_important]

#BASE MODELS
y_dnn_baseline = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_dnn_baseline.npy")
y_attention_lstm_baseline_mem1 = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_attention_lstm_baseline_mem1.npy")
y_lstm_baseline_mem1 = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_lstm_baseline_mem1.npy")
y_xgb_baseline = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_xgb_baseline.npy")
y_lr_baseline = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_lr_baseline.npy")

y_baseline = [y_dnn_baseline,y_attention_lstm_baseline_mem1,y_lstm_baseline_mem1,y_xgb_baseline,y_lr_baseline]

#WINDOW 24 FOR DAILY MOTIFS (default) AND 168 FOR WEEKLY MOTIFS
window_size=24

arr = np.arange(1,y_test.shape[1]+1)
weights = arr/y_test.shape[1] #24 for shape of y_test sequences


X_axis = np.arange(base_test_list_all[0].shape[0])
labels = np.arange(base_test_list_all[0].shape[0])

print(arr,weights)

"""
##DEMO OF FIRST AND LAST ELEMENT IN THE UNANCHORED CHAIN OF THE FIRST SEQUENCE
matrix_profile = stumpy.stump(y_test.values[:,0], m=window_size)
print(matrix_profile,matrix_profile.shape)
#GET ALL-CHAIN SET AND UNANCHORED CHAIN
all_chain_set, unanchored_chain = stumpy.allc(matrix_profile[:, 2], matrix_profile[:, 3])
print("ALL CHAINS")
print(all_chain_set,len(all_chain_set))
print("UNANCHORED_CHAIN")
print(unanchored_chain,unanchored_chain.shape)

#plotting some chain elements
pp.axis('off')
for i in range(unanchored_chain.shape[0]):
    data = y_test.iloc[unanchored_chain[i]:unanchored_chain[i]+window_size].reset_index().values
    x = data[:, 0]
    y = data[:, 1]
    pp.axvline(x=x[0]-x.min()+(window_size+5)*i + 11, alpha=0.3)
    pp.axvline(x=x[0]-x.min()+(window_size+5)*i + 15, alpha=0.3, linestyle='-.')
    pp.plot(x-x.min()+(window_size+5)*i, y-y.min(), linewidth=1)
pp.show()

#pp.plot(y_test, linewidth=1, color='black')
for i in range(unanchored_chain.shape[0]):
    y = y_test.values[unanchored_chain[i]:unanchored_chain[i]+window_size,0]
    pp.plot(y,label='Subsequence '+str(i))
    pp.xlabel("Timesteps")
    pp.ylabel("Load Subsequence Values (MW)")
    pp.title("Subsequence "+str(i))
    pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\chain_illustration_sub_'+str(i)+'.png', dpi=1000)
    pp.show()

#exit()

dtw_per_seq=[]

# FIRST ELEMENT IN THE CHAIN
y = y_test.values[unanchored_chain[0]:unanchored_chain[0]+window_size,0]
pp.plot(y,label='Actual')
for m,v in enumerate(y_cc):
    y_model=y_baseline[m][unanchored_chain[0]:unanchored_chain[0]+window_size,0]
    pp.plot(y_model,label=model_names[m]) #CC only for hybrid models
pp.xlabel("Timesteps")
pp.ylabel("Total Load (MW)")
pp.title("First 168-hour Subsequence")
pp.legend(fontsize=5,loc='best')
#pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\cc_example_chain_weekly_first.png', dpi=1000)
pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\baseline_example_chain_weekly_first.png', dpi=1000)
pp.show()

# LAST ELEMENT IN THE CHAIN
y = y_test.values[unanchored_chain[unanchored_chain.shape[0]-1]:unanchored_chain[0]+window_size,0]
pp.plot(y,label='Actual')
for m,v in enumerate(y_cc):
    y_model=y_baseline[m][unanchored_chain[unanchored_chain.shape[0]-1]:unanchored_chain[unanchored_chain.shape[0]-1]+window_size,0]
    pp.plot(y_model,label=model_names[m])
pp.xlabel("Timesteps")
pp.ylabel("Total Load (MW)")
pp.title("Last 168-hour Subsequence")
pp.legend(fontsize=5,loc='best')
#pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\cc_example_chain_weekly_last.png', dpi=1000)
pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\baseline_example_chain_weekly_last.png', dpi=1000)
pp.show()

print("END OF DEMO")

exit()
#END OF DEMO, COMPUTATION OF METRIC FOLLOWS
"""

"""
#ALL UNANCHORED CHAINS FOR SEQ k (METRIC CALCULATION)
wacc_array = []
dist_array = []
for m in range(len(y_cc)):
    avg_dtw_all=[]
    for k in range(y_test.shape[1]):
        matrix_profile = stumpy.stump(y_test.values[:,k], m=window_size)
        print(matrix_profile,matrix_profile.shape)
        #GET ALL-CHAIN SET AND UNANCHORED CHAIN
        all_chain_set, unanchored_chain = stumpy.allc(matrix_profile[:, 2], matrix_profile[:, 3])
        #print("ALL CHAINS")
        #print(all_chain_set,len(all_chain_set))
        print("UNANCHORED_CHAIN")
        print(unanchored_chain,unanchored_chain.shape)
        dtw_per_seq=[]
        for i in range(unanchored_chain.shape[0]):
            y = y_test.values[unanchored_chain[i]:unanchored_chain[i]+window_size,k]
            y_model=y_baseline[m][unanchored_chain[i]:unanchored_chain[i]+window_size,k]
            #dtw_score = calculate_dtw(y, y_model)
            dtw_score = euclidean_distance(y, y_model) #RUN IT WITH EUCLIDEAN DISTANCE
            print("DTW FOR CHAIN "+str(i)+": ",dtw_score)
            dtw_per_seq.append(dtw_score)
            pp.plot(y)
            pp.plot(y_model)
            pp.show()
            
        print("DTW LIST:",dtw_per_seq)
        avg_dtw_per_seq=find_average(dtw_per_seq)
        print(avg_dtw_per_seq)
        avg_dtw_all.append(avg_dtw_per_seq)
    print("AVERAGE DTWs FOR EACH SEQ ",avg_dtw_all)
    weighted_average_chain_closeness = np.average(np.array(avg_dtw_all),weights=np.flip(weights))
    print(weighted_average_chain_closeness)
    wacc_array.append(weighted_average_chain_closeness)
    dist_array.append(avg_dtw_all)
    

np_wacc = np.array(wacc_array)
np_dist = np.array(dist_array)

#y_cc, window_size=24    
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_wacc_scores.npy",np_wacc)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_distances.npy",np_dist)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_wacc_scores_euclidean.npy",np_wacc)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_distances_euclidean.npy",np_dist)

#y_cc, window_size=168    
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_wacc_scores_weekly.npy",np_wacc)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_distances_weekly.npy",np_dist)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_wacc_scores_weekly_euclidean.npy",np_wacc)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_distances_weekly_euclidean.npy",np_dist)

#y_baseline, window_size=24    
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_wacc_scores.npy",np_wacc)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_distances.npy",np_dist)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_wacc_scores_euclidean.npy",np_wacc)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_distances_euclidean.npy",np_dist)

#y_baseline, window_size=168    
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_wacc_scores_weekly.npy",np_wacc)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_distances_weekly.npy",np_dist)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_wacc_scores_weekly_euclidean.npy",np_wacc)
#np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_distances_weekly_euclidean.npy",np_dist)
"""
    

"""
### VISUALIZE WACC RESULTS --- DTW
#DAILY WACC CC
cc_wacc_daily=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_wacc_scores.npy")
cc_dist_daily=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_distances.npy")

print("DAILY WACC CC - DTW")
print(cc_wacc_daily)
print(cc_dist_daily)
print(cc_dist_daily[0,:])

for i in range(cc_dist_daily.shape[0]):
    pp.plot(np.flip(cc_dist_daily[i,:]),label='CC-'+model_names[i])

pp.xticks(X_axis)
pp.xlabel("Predicted Sequence")
pp.ylabel("Average DTW Score")
pp.title("Average Chain Similarity On 24-Hour Subsequences")
pp.legend()
#pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\cc_average_daily_chain_similarity.png', dpi=1000)
pp.show()


#WEEKLY WACC CC
cc_wacc_weekly=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_wacc_scores_weekly.npy")
cc_dist_weekly=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_distances_weekly.npy")

print("WEEKLY WACC CC - DTW")
print(cc_wacc_weekly)
print(cc_dist_weekly)
print(cc_dist_weekly[0,:])

for i in range(cc_dist_weekly.shape[0]):
    pp.plot(np.flip(cc_dist_weekly[i,:]),label='CC-'+model_names[i])

pp.xticks(X_axis)
pp.xlabel("Predicted Sequence")
pp.ylabel("Average DTW Score")
pp.title("Average Chain Similarity On 168-Hour Subsequences")
pp.legend()
#pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\cc_average_weekly_chain_similarity.png', dpi=1000)
pp.show()

##BASELINE MODEL WACC
#DAILY WACC BASE
base_wacc_daily=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_wacc_scores.npy")
base_dist_daily=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_distances.npy")

print("DAILY WACC BASELINE - DTW")
print(base_wacc_daily)
print(base_dist_daily)
print(base_dist_daily[0,:])

for i in range(base_dist_daily.shape[0]):
    pp.plot(np.flip(base_dist_daily[i,:]),label=model_names[i])

pp.xticks(X_axis)
pp.xlabel("Predicted Sequence")
pp.ylabel("Average DTW Score")
pp.title("Average Chain Similarity On 24-Hour Subsequences")
pp.legend()
#pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\base_average_daily_chain_similarity.png', dpi=1000)
pp.show()

#WEEKLY WACC BASE
base_wacc_weekly=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_wacc_scores_weekly.npy")
base_dist_weekly=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_distances_weekly.npy")

print("WEEKLY WACC BASELINE-DTW")
print(base_wacc_weekly)
print(base_dist_weekly)
print(base_dist_weekly[0,:])

for i in range(base_dist_weekly.shape[0]):
    pp.plot(np.flip(base_dist_weekly[i,:]),label=model_names[i])

pp.xticks(X_axis)
pp.xlabel("Predicted Sequence")
pp.ylabel("Average DTW Score")
pp.title("Average Chain Similarity On 168-Hour Subsequences")
pp.legend()
#pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\base_average_weekly_chain_similarity.png', dpi=1000)
pp.show()
"""


###########
### VISUALIZE WACC RESULTS --- EUCLIDEAN
#DAILY WACC CC
cc_wacc_daily=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_wacc_scores_euclidean.npy")
cc_dist_daily=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_distances_euclidean.npy")

print("DAILY WACC CC-EUCLIDEAN")
print(cc_wacc_daily)
print(cc_dist_daily)
print(cc_dist_daily[0,:])

for i in range(cc_dist_daily.shape[0]):
    pp.plot(np.flip(cc_dist_daily[i,:]),label='CC-'+model_names[i])

pp.xticks(X_axis)
pp.xlabel("Predicted Sequence")
pp.ylabel("Average Euclidean Distance")
pp.title("Average Chain Similarity On 24-Hour Subsequences")
pp.legend()
#pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\cc_average_daily_chain_similarity_euclidean.png', dpi=1000)
pp.show()

#WEEKLY WACC CC
cc_wacc_weekly=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_wacc_scores_weekly_euclidean.npy")
cc_dist_weekly=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\cc_models_distances_weekly_euclidean.npy")

print("WEEKLY WACC CC-EUCLIDEAN")
print(cc_wacc_weekly)
print(cc_dist_weekly)
print(cc_dist_weekly[0,:])

for i in range(cc_dist_weekly.shape[0]):
    pp.plot(np.flip(cc_dist_weekly[i,:]),label='CC-'+model_names[i])

pp.xticks(X_axis)
pp.xlabel("Predicted Sequence")
pp.ylabel("Average Euclidean Distance")
pp.title("Average Chain Similarity On 168-Hour Subsequences")
pp.legend()
#pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\cc_average_weekly_chain_similarity_euclidean.png', dpi=1000)
pp.show()

##BASELINE MODEL WACC
#DAILY WACC BASE
base_wacc_daily=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_wacc_scores_euclidean.npy")
base_dist_daily=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_distances_euclidean.npy")

print("DAILY WACC BASELINE-EUCLIDEAN")
print(base_wacc_daily)
print(base_dist_daily)
print(base_dist_daily[0,:])

for i in range(base_dist_daily.shape[0]):
    pp.plot(np.flip(base_dist_daily[i,:]),label=model_names[i])

pp.xticks(X_axis)
pp.xlabel("Predicted Sequence")
pp.ylabel("Average Euclidean Distance")
pp.title("Average Chain Similarity On 24-Hour Subsequences")
pp.legend()
#pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\base_average_daily_chain_similarity_euclidean.png', dpi=1000)
pp.show()

#WEEKLY WACC BASE
base_wacc_weekly=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_wacc_scores_weekly_euclidean.npy")
base_dist_weekly=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\motifs\\base_models_distances_weekly_euclidean.npy")

print("WEEKLY WACC BASELINE-EUCLIDEAN")
print(base_wacc_weekly)
print(base_dist_weekly)
print(base_dist_weekly[0,:])

for i in range(base_dist_weekly.shape[0]):
    pp.plot(np.flip(base_dist_weekly[i,:]),label=model_names[i])

pp.xticks(X_axis)
pp.xlabel("Predicted Sequence")
pp.ylabel("Average Euclidean Distance")
pp.title("Average Chain Similarity On 168-Hour Subsequences")
pp.legend()
#pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\base_average_weekly_chain_similarity_euclidean.png', dpi=1000)
pp.show()




"""
#VALIDATE STATIONARITY OF INPUT
print(X_train_norm,X_train_norm.shape)
test_statistics=[]
p_values=[]
crit_1=[]
crit_5=[]
crit_10=[]
for i in range(X_train_norm.shape[1]):
    adfuller_results = adfuller(X_train_norm[:,i])
    print('Feature ',i)
    print(adfuller_results[0],adfuller_results[1],adfuller_results[4].get('1%'),adfuller_results[4].get('5%'),adfuller_results[4].get('10%'))
    test_statistics.append(adfuller_results[0])
    p_values.append(adfuller_results[1])
    crit_1.append(adfuller_results[4].get('1%'))
    crit_5.append(adfuller_results[4].get('5%'))
    crit_10.append(adfuller_results[4].get('10%'))


np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\stationarity\\test_statistics.npy",np.array(test_statistics))
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\stationarity\\p_values.npy",np.array(p_values))
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\stationarity\\crit_1.npy",np.array(crit_1))
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\stationarity\\crit_5.npy",np.array(crit_5))
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\stationarity\\crit_10.npy",np.array(crit_10))
"""
"""
#VISUALIZE STATIONARITY
test_stats = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\stationarity\\test_statistics.npy")
p_vals = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\stationarity\\p_values.npy")
cr_1 = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\stationarity\\crit_1.npy")
cr_5 = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\stationarity\\crit_5.npy")
cr_10 = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\stationarity\\crit_10.npy")

#CRITICAL VALUE ASSESSMENT
pp.plot(test_stats,label='Test Statistic')
pp.plot(cr_1,label='Critical Value at 1%')
pp.plot(cr_5,label='Critical Value at 5%')
pp.plot(cr_10,label='Critical Value at 10%')
pp.xlabel('Input Features')
pp.ylabel('Augmented Dickey-Fuller Statistic Values')
pp.title('Critical Value Assessment')
pp.legend()
pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\critical_value_assessment.png', dpi=1000)
pp.show()

#P-VALUE ASSESSMENT
#pp.plot(0.05*np.ones(shape=X_train_norm.shape[1]),label='p-value threshold')
pp.plot(p_vals)
pp.xlabel('Input Features')
pp.ylabel('p-values')
pp.title('p-value Assessment')
#pp.legend()
pp.savefig(r'D:Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\p-value_assessment.png', dpi=1000)
pp.show()
"""

"""
##LOAD NEURAL NETWORK H5 MODELS TO VISUALIZE THEIR ARCHITECTURE
#CC MODELS
imp_dnn = load_model(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\dnn_dnn_important_components.h5')
imp_att_lstm = load_model(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\lstm_attention_lstm_mem1_important_components.h5')
imp_lstm = load_model(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\lstm_lstm_mem1_important_components.h5')
keras_models_imp=[imp_dnn,imp_att_lstm,imp_lstm]
keras_model_names_imp = ['imp_dnn','imp_att_lstm','imp_lstm']
for k_ind,keras_model in enumerate(keras_models_imp):
    plot_model(keras_model, to_file='D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\'+keras_model_names_imp[k_ind]+'.png',show_shapes=True,show_dtype=True,show_layer_activations=True,show_layer_names=False,dpi=1000)

#BASE MODELS
b_dnn = load_model(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\dnn_4000eps.h5')
b_att_lstm = load_model(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\lstm_attention_lstm_baseline_mem1.h5')
b_lstm = load_model(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\lstm_lstm_baseline_mem1.h5')
keras_models_imp=[b_dnn,b_att_lstm,b_lstm]
keras_model_names_imp = ['base_dnn','base_att_lstm','base_lstm']
for k_ind,keras_model in enumerate(keras_models_imp):
    plot_model(keras_model, to_file='D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\charts\\'+keras_model_names_imp[k_ind]+'.png',show_shapes=True,show_dtype=True,show_layer_activations=True,show_layer_names=False,dpi=1000)
"""


"""
#LOAD THE SAVED BASELINE MLP DNN AND EVALUATE (EXPERIMENTS ON OUTPUTS)
base_dnn = load_model(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\dnn_4000eps.h5')
y_pred_test=base_dnn.predict(X_test_norm)
y_pred_train=base_dnn.predict(X_train_norm)
print(y_pred_test,y_pred_test.shape)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\y_dnn_baseline.npy",y_pred_test)
eval_multiseq(y_train, y_pred_train,name = 'dnn_baseline_train')


eval_multiseq(y_test, y_pred_test,name = 'dnn_baseline')
#GET WEIGHTS
dnn_weights=base_dnn.get_weights()
#print(dnn_weights,len(dnn_weights)) # model has 3 layers, getting 3 weight and 3 bias tables
#print(dnn_weights[0],dnn_weights[0].shape)
#print(dnn_weights[1],dnn_weights[1].shape)
#print(dnn_weights[2],dnn_weights[2].shape)
#print(dnn_weights[3],dnn_weights[3].shape)
#print(dnn_weights[4],dnn_weights[4].shape)
#print(dnn_weights[5],dnn_weights[5].shape)
#print("SHAPES OF WEIGHTS AND BIASES: ")
#print(dnn_weights[0].shape,dnn_weights[1].shape,dnn_weights[2].shape,dnn_weights[3].shape,dnn_weights[4].shape,dnn_weights[5].shape) #GET SHAPE STRUCTURE FOR WEIGHT AND BIAS MATRICES

#ANALYZE WEIGHTS OF A LAYER
#print(dnn_weights[2],dnn_weights[2].shape)
#print(dnn_weights[0][0,:],dnn_weights[0][0,:].shape)

N=3
res_list=[]
res_freqs={}

l_out= [[] for _ in range(dnn_weights[4].shape[1])] # containers of best inputs for each output
#print(l_out, l_out[0])

for j in range(dnn_weights[4].shape[1]):
#    print(dnn_weights[2][j],dnn_weights[2][j].shape)
    #pp.plot(dnn_weights[0][:,j],label='Neuron '+str(j)) # [:,i] for ith sequence
    #pp.legend()
    #pp.show()
    neuron_weights = dnn_weights[4][:,j]
    #DISTANCES BETWEEN NEURONS
    #eu_dis,man_dis,cos_dis = similarity_metrics(dnn_weights[0][:,0], dnn_weights[0][:,j])
    #print(eu_dis,man_dis,cos_dis)
    #N-LARGEST INPUTS PER NEURON N=5
    res = sorted(range(len(neuron_weights)), key = lambda sub: neuron_weights[sub])[-N:]
    #print("Top 3 inputs in OUTPUT neuron "+str(j) +" : "+ str(res)) #GET TOP 3 FOR EACH OUTPUT LAYER NEURON
    res_list = res_list+res
    
    inp_list=[]
    
    d2_list=[]
    for m in res:
        neuron_weights_d2 = dnn_weights[2][:,m]
        
        res_d2 = sorted(range(len(neuron_weights_d2)), key = lambda sub: neuron_weights_d2[sub])[-N:]
        #print("Top 3 inputs in D2 neuron "+str(m) +" : "+ str(res_d2)) #GET TOP 3 FOR EACH D2 LAYER NEURON
        d2_list =d2_list+res_d2
        
        for n in res_d2:
            neuron_weights_d1 = dnn_weights[0][:,n]
                    
            res_d1 = sorted(range(len(neuron_weights_d1)), key = lambda sub: neuron_weights_d1[sub])[-N:]
            #print("Top 3 inputs in D1 neuron "+str(n) +" : "+ str(res_d1)) #GET TOP 3 FOR EACH D1 LAYER NEURON
            inp_list = inp_list + res_d1
    
    #print(len(inp_list))
    #print(inp_list)
    l_out[j].append(inp_list)

#for i in range(dnn_weights[4].shape[1]):
    #print("INPUTS STRONGLY INVOLVED IN THE INFERENCE OF OUTPUT ",i)
    #print(l_out[i][0]) #second 0 to get the list with simple brackets
    #print(list(set(inp_list)))
    #exit()

#GET SPECIFIC COLUMNS FROM X_TRAIN
#print("TARGETED INPUT")
#print(X_train_norm[:,l_out[0][0]],X_train_norm[:,l_out[0][0]].shape)
#print(y_train.values,y_train.values.shape)
#print(y_train.values[:,0],y_train.values[:,0].shape, )


##IDEA 1: NEURAL SUBGROUPS AS DENOISED ENCODED VECTORS
#BUILD MODEL WITH EACH SET OF INVOLVED FEATURES
for i in range(dnn_weights[4].shape[1]):
    
    f_nodupes=list(set(l_out[i][0]))
    #X_TRAIN,X_TEST
    X_tr=X_train_norm[:,f_nodupes]
    X_te=X_test_norm[:,f_nodupes]
    #Y_TRAIN,Y_TEST
    y_tr = y_train.values[:,i]
    y_te = y_test.values[:,i]
    
    #CREATE ATTENTION VECTORS FOR EACH FEATURE SET
    
    #COMPRESS INPUTS THROUGH AUTOENCODER
    #run_name='autoencoder_lstm_'+str(i)
    #y_pred_test,y_pred_train, evalX, evalY = LSTM_run(trainset_X=X_tr,trainset_Y=y_tr,testset_X=X_te,testset_Y=y_te, lstm_units=100,n_out=y_te.ndim,eps=20,bsize=72, name=run_name)
    #print(y_pred_test,y_pred_test.shape)
    #print(y_pred_train,y_pred_train.shape)
    
    y_hat = naive_lr(X_tr, y_tr, X_te)
    pp.plot(y_hat,label='Predicted') # [:,i] for ith sequence
    pp.plot(y_te,label='Actual')
    pp.legend()
    pp.show()
    
    #print("PREDICTION FOR SEQUENCE "+str(i)+" BASED ON INVOLVED FEATURES")
    #eval_multiseq(y_te, y_hat)

#LOAD AUTOENCODER REPRESENTATION FOR SEQ T=0 (TRY AND CONCATENATE ALL IN A LOOP)
X_tr = X_train_norm[25:,:]
X_te = X_test_norm[25:,:]
for k in range(dnn_weights[4].shape[1]):
    X_tr_enc = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\X_train_autoencoder_lstm_"+str(k)+"_encoded.npy")
    X_te_enc = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\X_test_autoencoder_lstm_"+str(k)+"_encoded.npy")
    
    for m in range(X_tr_enc.ndim):
        X_tr_den = fft_denoiser(X_tr_enc[:,m],10, True,name='X_tr_'+str(k)+'_'+str(m))
        X_te_den = fft_denoiser(X_te_enc[:,m],10, True,name='X_te_'+str(k)+'_'+str(m))
        
        #pp.plot(X_tr_den,label='Encoded')
        #pp.show()
        
        #print(X_tr_den,X_tr_den[:,None].shape) #INCLUDE DENOISED FEATURES 
        X_tr = np.concatenate((X_tr,X_tr_den[:,None]),axis=1)
        X_te = np.concatenate((X_te,X_te_den[:,None]),axis=1)
    
    #CONCATENATE IF NOT DENOISING
    #X_tr = np.concatenate((X_tr,X_tr_enc),axis=1)
    #X_te = np.concatenate((X_te,X_te_enc),axis=1)
    
    #pp.plot(X_tr_enc,label='Encoded')
    #pp.show()


#print(X_tr.shape,X_te.shape)
#print(y_train.values[25:,0].shape)

y_hat = naive_lr(X_tr, y_train.values[25:,0], X_te)
pp.plot(y_hat,label='Predicted') # [:,i] for ith sequence
pp.plot(y_te,label='Actual')
pp.legend()
pp.show()

#print("PREDICTION FOR SEQUENCE "+"0"+" BASED ON INVOLVED FEATURES")
#eval_multiseq(y_test.values[25:,0], y_hat)
#END OF IDEA #1

##IDEA 2: ROBUST PCA ON FEATURE SUBGROUPS
for i in range(dnn_weights[4].shape[1]):
    
    f_nodupes=list(set(l_out[i][0]))
    print(len(f_nodupes))
    
    #X_TRAIN,X_TEST
    X_tr=X_train_norm[:,f_nodupes]
    X_te=X_test_norm[:,f_nodupes]
    #Y_TRAIN,Y_TEST
    y_tr = y_train.values[:,i]
    y_te = y_test.values[:,i]
    
    for k in range(len(f_nodupes)):
        pp.plot(X_tr[0:300,k],label='feature '+str(k))
        pp.legend()
        pp.show()
    
    exit()


###EXPERIMENTAL MODELS ON KERNEL FEATURE SET

##MODEL #1: CDEN-LSTM
run_name='cden_lstm'
y_pred_test,y_pred_train, evalX, evalY_test, evalY_train = LSTM_run(trainset_X=X_tr,trainset_Y=y_train.values[25:,:],testset_X=X_te,testset_Y=y_test.values[25:,:], lstm_units=100,n_out=24,eps=4000,bsize=72, name=run_name)
print(y_pred_test,y_pred_test.shape)
print(y_pred_train,y_pred_train.shape)

#EVALUATE TRAINING PERFORMANCE
print("TRAINING PERFORMANCE")
eval_multiseq(evalY_train, y_pred_train, name = run_name+'_train')
#EVALUATE TESTING PERFORMANCE
print("TESTING PERFORMANCE")
eval_multiseq(evalY_test, y_pred_test, name = run_name+'_test')

results_test=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\metrics_"+run_name+"_test_model.npy")
print(results_test)

##MODEL #2: CDEN-DNN
run_name='dnn_expanded'
#y_pred_test,y_pred_train = DNN_run(trainset_X=X_train_norm,trainset_Y=y_train.values,testset_X=X_test_norm,testset_Y=y_test.values, neurons_h1 = X_tr.shape[1], neurons_h2=X_tr.shape[1]//2, neurons_out = y_train.shape[1],input_dimension =  X_train_norm.shape[1], bsize=72, eps=4000, name=run_name)
#print(y_pred_test,y_pred_test.shape)
#print(y_pred_train,y_pred_train.shape)

expanded_dnn = load_model(r'D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\dnn_dnn_expanded.h5')
y_pred_test=expanded_dnn.predict(X_test_norm)
#print(y_pred_test,y_pred_test.shape)
eval_multiseq(y_test, y_pred_test,name = 'dnn_expanded')

#EVALUATE TRAINING PERFORMANCE
print("TRAINING PERFORMANCE")
#eval_multiseq(y_train.values[25:,:], y_pred_train, name = run_name+'_train')
#EVALUATE TESTING PERFORMANCE
print("TESTING PERFORMANCE")
#eval_multiseq(y_test.values[25:,:], y_pred_test, name = run_name+'_test')

results_test=np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\metrics\metrics_"+run_name+"_test_model.npy")
print(results_test)

exit()

print(res_list)
for item in res_list:
   # checking the element in dictionary
   if item in res_freqs:
      # incrementing the counr
      res_freqs[item] += 1
   else:
      # initializing the count
      res_freqs[item] = 1

print(res_freqs)
sorted_freqs = dict(sorted(res_freqs.items(), key=operator.itemgetter(1))) #SORT INPUT FREQS, LAST N ARE THE MOST FREQUENT
print(sorted_freqs)
n_freqs = list(sorted_freqs)[len(sorted_freqs)-5:len(sorted_freqs)] #N most frequend ascending
n_freqs.reverse() #N most frequent descending
print(n_freqs)
#end of weight analysis
"""

"""
###(OPTIONALLY) RESTRUCTURE MAIN DATASET FOR MULTIPLE FEATURES PER TIMESTEP (LSTM BLOCK)
load_laglist=X_train.columns[pd.Series(X_train.columns).str.startswith('GR_load_actual_entsoe_transparencylag_')].tolist()
load_lagnum=[]
solar_laglist=[]
for l in load_laglist:
    lnum=l.replace('GR_load_actual_entsoe_transparencylag_','')
    s_lag = l.replace('GR_load_actual_entsoe_transparencylag_','GR_solar_generation_actuallag_')
    #print(lnum)
    load_lagnum.append(lnum)
    solar_laglist.append(s_lag)
#print(load_laglist,solar_laglist)

X_lstm = d_main[load_laglist+solar_laglist]
y_lstm = d_main[corr_df['targets'].dropna().values]
#print(X_lstm)

###SPLIT LSTM DATA INTO TRAIN AND TEST SET AND CREATE 3D FORMAT
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.20,shuffle = False)
#print(X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm)
#print(X_train_lstm.shape, X_test_lstm.shape, y_train_lstm.shape, y_test_lstm.shape)


###NORMALIZE LSTM DATASET
scaler_lstm = MinMaxScaler()
X_train_lstm_norm = scaler_lstm.fit_transform(X_train_lstm)
X_test_lstm_norm = scaler_lstm.transform(X_test_lstm)
df_train_lstm = pd.DataFrame(X_train_lstm_norm, columns=X_train_lstm.columns)
df_test_lstm = pd.DataFrame(X_test_lstm_norm, columns=X_test_lstm.columns)
"""

"""
###CONVERT NORMALIZED DATA TO 3D FORMAT
values_struct_train = np.empty([X_train_lstm.shape[0],len(load_lagnum),2])
values_struct_test = np.empty([X_test_lstm.shape[0],len(load_lagnum),2])
for c,n in enumerate(load_lagnum):
    #PREPARE 3D TRAIN SET
    for i in range(X_train_lstm.shape[0]):
        values_struct_train[i][c][0]=df_train_lstm['GR_load_actual_entsoe_transparencylag_'+n].values[i]
        values_struct_train[i][c][1]=df_train_lstm['GR_solar_generation_actuallag_'+n].values[i]
    #PREPARE 3D TEST SET
    for j in range(X_test_lstm.shape[0]):
        values_struct_test[j][c][0]=df_test_lstm['GR_load_actual_entsoe_transparencylag_'+n].values[j]
        values_struct_test[j][c][1]=df_test_lstm['GR_solar_generation_actuallag_'+n].values[j]
        

print(values_struct_train, values_struct_train.shape, values_struct_test, values_struct_test.shape) #3D ARRAYS OF CORRELATED LAGS + SOLAR GENERATION INFO
#TO DO: SAVE THOSE ARRAYS TO LOAD THEM WHEN NEEDED
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\X_train_lstm.npy",values_struct_train)
np.save("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\X_test_lstm.npy",values_struct_test)
### END OF 3D RESTRUCTURING
"""


"""
#LOAD AND PASS THEM TO THE SIMPLE LSTM MODEL THAT WILL BE MADE A FUNCTION (NEEDS CODE FROM OPTIONALLY TO NORMALIZE)

X_train_struct = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\X_train_lstm.npy")
X_test_struct = np.load("D:\Post-Doc_Research\Datasets\open_power_system\opsd-time_series-2020-10-06\X_test_lstm.npy")
#print(X_train_struct, X_train_struct.shape, X_test_struct, X_test_struct.shape)

y_pred = lstm_network(X_train_struct,X_test_struct,y_train_lstm,timesteps=X_train_struct.shape[1],features=X_train_struct.shape[2],lstm_units=64,dense_units=50,output_units=24,passes=10)
"""

"""
#CREATE SIMPLE MULTI-STEP LSTM
X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

#print(X_train.values[0])
#print(X_train_lstm[0,0,:],X_train_lstm.shape)

print(X_train_lstm.shape,y_train.shape,X_test_lstm.shape,y_test.shape)


# define model (SIMPLE LSTM that works)
n_steps_out = 24
model = Sequential()
model.add(InputLayer((X_train_lstm.shape[1],X_train_lstm.shape[2])))
model.add(LSTM(64))
model.add(Dense(50,'relu'))
model.add(Dense(24,'linear'))
model.compile(optimizer='adam', loss= 'mae')

model.summary()
model.fit(X_train_lstm, y_train.values, epochs = 10, verbose=1)
yhat = model.predict(X_test_lstm)
print(yhat,yhat.shape)
"""

#END OF CODE COMMAND TO CLEAR TENSORFLOW SESSION
tf.keras.backend.clear_session()