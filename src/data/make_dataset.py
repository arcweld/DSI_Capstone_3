from scipy.io import loadmat
import pandas as pd
import numpy as np
import os
import sys
import logging

DATA_PATH = os.path.join('..','..','data','raw')

datafile = 'sat-6-full.mat'
data = loadmat(os.path.join(DATA_PATH, datafile))


if __name__ == "__main__":
    load_data()
