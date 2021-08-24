import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn as skl
from sklearn import preprocessing
from scipy import ndimage

# changes data into Pytorch tensors, so they can be used for models
def createTrainTensors(train):
    # Train data
    np_train_data = train.to_numpy()
    x = torch.from_numpy(np_train_data.astype(np.float32))
    return x


def createParamsTensors(train, params):
    # Params data
    np_params_data = params.to_numpy()
    np_params_data_reshape = np_params_data.reshape(len(train.T), 1)
    # np_params_data_reshape = np_params_data.reshape(500, 55,1)

    z = torch.from_numpy(np_params_data_reshape)  # .astype)#(np.float32))

    return z


def createParamsNumpy(train, params):
    np_params_data = params.to_numpy()
    y = np_params_data.reshape(len(train.T), 1)
    # y = np_params_data.reshape(500, 55 ,1)

    return y


def createTestTensors(test):
    # Test data
    if len(test) > 0:
        np_test_data = test.to_numpy()
        x_test = torch.from_numpy(np_test_data.astype(np.float32))


#Calculates median for dataframe
def calculateMedianValue(data):

    #medianData = data.T
    #medianData = medianData.rolling(3).median()
    #medianData = medianData.T
    #medianData = medianData.fillna(1)

    medianData = ndimage.median_filter(data, 3)
    return medianData

#first 30 values are replaced by the value of 1, based on the baseline solution
def replaceFirstXValues(data_frame, to_replace):
    for i in range(to_replace):
        data_frame[i] = 1
    return data_frame

#removes first x values from dataframe
def removeFirstXValues(data, to_remove):
    print(data.shape)
    if type(data) != np.ndarray:
        for i in range(to_remove):
            data = data.drop(labels=i, axis=1)

    else:
        data = data[:,to_remove:]

    return data

def replaceValuesOverOne(data_frame):
    print(type(data_frame))
    cleaned_list = []
    for element in data_frame:
        for value in element:
            if value > 1:
                cleaned_list.append(1)
            else:
                cleaned_list.append(value)
    np_cleaned = np.array(cleaned_list)
    np_cleaned = np_cleaned.reshape(len(np_cleaned), 1)
    print(type(np_cleaned))
    return np_cleaned

def zScoreDataFrame(data_frame, mean_value, standard_deviation):

    if type(data_frame) != np.ndarray:
       data_frame = data_frame.to_numpy()

    #mean_value = np.mean(data_frame)
    #standard_deviation = np.std(data_frame)

    scaled_data_frame = (data_frame - mean_value) / standard_deviation
    print(f"Rescaling data with mean = {mean_value}, std = {standard_deviation}")
    return scaled_data_frame

def rescaleData(data_frame, mean_value, standard_deviation):
    data_unscaled = data_frame * standard_deviation + mean_value
    return data_unscaled

#normalizes values for each measurement(1-300)
def normalizeDataFrame(data_frame):

    values = data_frame  # .values
    values = values.T
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(values)
    scaled_values = scaled_values.T

    norm_data = pd.DataFrame(scaled_values)
    return norm_data

def preprocessDataTorch(data_frame):
    print("skrrrt")
    data_frame = data_frame.numpy()

    median = False
    replaceValuesBiggerOne = False
    removeFirstValues = True
    zScore = True

    replaceFirstXValues = False #alternative --> removeFirstValues
    normalizeData = False #alternative --> zScore


    # train
    mean_value = 0.9940023784109137  # np.mean(train_data)
    standard_deviation = 0.03675546640117229  # np.std(train_data)

    # target --> unscaled atm
    mean_value_target = 0.055243712320442104  # np.mean(params_data)
    standard_deviation_target = 0.03576535986430257  # np.std(params_data)

    #parameters to declare
    to_replace = 10
    to_remove = 10

    if median == True:
        data_frame = calculateMedianValue(data_frame)

    if replaceValuesBiggerOne == True:
        print("hi")
        data_frame = replaceValuesOverOne(data_frame)

    if removeFirstValues == True:
        data_frame = removeFirstXValues(data_frame, to_remove)

    if zScore == True:
        data_frame = zScoreDataFrame(data_frame, mean_value, standard_deviation)

    data_frame = torch.from_numpy(data_frame.astype(np.float32))
    return data_frame

    #if normalizeData == True:
        #data_frame = normalizeDataFrame(data_frame)

    #if replaceFirstValues == True:
        #data_frame = replaceFirstXValues(data_frame)


def preprocessParams(data_frame):
    data_frame = normalizeDataFrame(data_frame)
    return data_frame