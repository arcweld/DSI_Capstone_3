#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.io import loadmat
import pandas as pd
import numpy as np
import os
import sys
import logging

module_path = os.path.abspath(os.path.join('..','data'))
if module_path not in sys.path:
    sys.path.append(module_path)
DATA_PATH = os.path.join('data','raw')


def fetch_training_data(data_path=DATA_PATH, nrows=None, chunksize=2000):
    X_tr_datafile = data_path + '/X_train_sat6.csv'
    X_tr = mem_managed_df_read(X_tr_datafile, nrows, chunksize)
    y_tr_datafile = data_path + '/y_train_sat6.csv'
    y_tr = pd.read_csv(y_tr_datafile, nrows=nrows, header=None)
    return X_tr, y_tr

def fetch_testing_data(data_path=DATA_PATH, nrows=None, chunksize=2000):
    X_test_datafile = data_path + '/X_test_sat6.csv'
    X_test = mem_managed_df_read(X_test_datafile, nrows, chunksize)
    y_test_datafile = data_path + '/y_test_sat6.csv'
    y_test = pd.read_csv(y_test_datafile, nrows=nrows, header=None)
    return X_test, y_test

def mem_managed_df_read(datafile, nrows=None, chunksize=2000):
    X = pd.DataFrame([])
    iter = 0
    total_iter = 32400 // chunksize
    for df in pd.read_csv(datafile, iterator=True, chunksize=chunksize, nrows=nrows, low_memory=False, header=None):
        X = X.append(df/255.)
        logging.debug('iter %s',str(iter))
        iter += 1
    return X

def labels_dictionary(data_path=DATA_PATH):
    labels = pd.read_csv(data_path + '/sat6annotations.csv', header=None).iloc[:,0].values
    labels_dict = {}
    for i in range(len(labels)):
      labels_dict[i] = labels[i]
    return labels_dict

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(fetch_training_data(nrows=5000))
