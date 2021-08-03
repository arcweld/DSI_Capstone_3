from scipy.io import loadmat
import pandas as pd
import numpy as np
import os
import sys
import logging

DATA_PATH = os.path.join('data','raw')
datafile = DATA_PATH + '/X_train_sat6.csv'

X_tr = pd.DataFrame([])
for df in pd.read_csv(datafile, iterator=True, chunksize=5000):
    X_tr = X_tr.append(df)
    print('*', end=' ')


if __name__ == "__main__":
    print(df.head())
