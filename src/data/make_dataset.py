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
for df in pd.read_csv(datafile, iterator=True, chunksize=200, low_memory=False):
    X_tr = X_tr.append(df)
    print(f'iter {iter} / 1620')
    logging.debug(f'appended, iteration {iter} = {200*(iter+1)} rows')
    iter += 1


if __name__ == "__main__":
    logging.basicConfig(filename='./app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    print(df.head())
