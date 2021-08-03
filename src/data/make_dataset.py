from scipy.io import loadmat
import pandas as pd
import numpy as np
import os
import sys
import logging

DATA_PATH = os.path.join('data','raw')
datafile = DATA_PATH + '/X_train_sat6.csv'

df = pd.read_csv(datafile)


if __name__ == "__main__":
    print(df.head())
