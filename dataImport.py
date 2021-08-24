import pandas as pd
import numpy as np


def importTrainingData(files, stop_read, count):
    # Import Training files
    if count%2 == 0:
        print("Importing Training Data...", "Iteration ", count)
    else:
        print("Importing Validation Data...", "Iteration ", count-1)
    training_list = []

    # Add first n files to list
    for filename in files:
        df = pd.read_csv(filename, header=None, sep="\t", skiprows=6)
        training_list.append(df)
        count = count + 1
        if count == stop_read:
            break

    # Convert List content to Dataframe
    train_data = pd.concat(training_list, axis=0, ignore_index=True)

    return train_data


def importParamsData(files, stop_read, count):
    # Import Params
    if count % 2 == 0:
        print("Importing Training Params...", "Iteration ", count)
    else:
        print("Importing Validation Params...", "Iteration ", count - 1)
    params_list = []

    # Add first n files to list
    for filename in files:
        df = pd.read_csv(filename, header=None, sep="\t", skiprows=2)
        params_list.append(df)
        count = count + 1
        if count == stop_read:
            break

    # Convert List content to Dataframe
    params_data = pd.concat(params_list, axis=0, ignore_index=True)

    return params_data

def importTestData(test_files):
    li = []

    for filename in test_files:
        df = pd.read_csv(filename, header=None, sep="\t", skiprows=6)
        li.append(df)

    test_frame = pd.concat(li, axis=0, ignore_index=True)

    return test_frame

def importTargetParameter(files, stop_read, count):
    train_parameters = []
    # Add first n files to list
    for filename in files:
        df = pd.read_csv(filename, header=None, sep=" ", nrows=2)
        train_parameters.append(df)
        count = count + 1
        if count == stop_read:
            break

    # Convert List content to Dataframe
    parameter_data = pd.concat(train_parameters, axis=0, ignore_index=True)

    return parameter_data


def importTrainParameter(files, stop_read, count):
    train_parameters = []

    # Add first n files to list
    for filename in files:
        df = pd.read_csv(filename, header=None, sep=" ", nrows=6)
        for x in range(55):
            train_parameters.append(df)
        count = count + 1
        if count == stop_read:
            break

    # Convert List content to Dataframe
    parameter_data = pd.concat(train_parameters, axis=0, ignore_index=True)

    return parameter_data

def importTrainParameterSingle(files, stop_read, count):
    train_parameters = []

    # Add first n files to list
    for filename in files:
        df = pd.read_csv(filename, header=None, sep=" ", nrows=6)
        #for x in range(55):
        train_parameters.append(df)
        count = count + 1
        if count == stop_read:
            break

    # Convert List content to Dataframe
    parameter_data = pd.concat(train_parameters, axis=0, ignore_index=True)

    return parameter_data

def importEvalParameter(files):
    eval_parameters = []

    # Add first n files to list
    for filename in files:
        df = pd.read_csv(filename, header=None, sep=" ", nrows=6)
        for x in range(55):
            eval_parameters.append(df)

    # Convert List content to Dataframe
    parameter_data = pd.concat(eval_parameters, axis=0, ignore_index=True)

    return parameter_data