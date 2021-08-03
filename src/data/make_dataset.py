from scipy.io import loadmat
import pandas as pd
import numpy as np
import os
import sys
import logging


DATA_PATH = os.path.join('data','raw')
datafile = DATA_PATH + '/X_train_sat6.csv'

iter = 0
X_tr = pd.DataFrame([])
for df in pd.read_csv(datafile, iterator=True, chunksize=2000, low_memory=False, header=None):
    X_tr = X_tr.append(df)
    print(f'iter {iter} / 162')
    logging.debug(f'appended, iteration {iter} = {2000*(iter+1)} rows')
    iter += 1


if __name__ == "__main__":
    logging.basicConfig(filename='./app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    print(df.head())
