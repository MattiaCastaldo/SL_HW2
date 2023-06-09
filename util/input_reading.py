import pandas as pd
import numpy as np

def to_matrix(val, features_num = 116, timestamp_num = 115):
    '''turns a row of a df into a matrix:
         the matrix has as columns the features (brain regions)
         and as rows the timestamps
    '''
    return val.values.reshape(features_num, timestamp_num).astype(float)

# Usage:
# df = pd.read_csv("PATH_TO_FILE")
# df.iloc[:, 4:].apply(to_matrix, axis=1) # and then .tolist() or assign to df column

def to_corr_matrix(val):
    '''turns a row of a df into a corr matrix between the features'''
    m = to_matrix(val)
    with np.errstate(invalid='ignore'):
        cor = np.corrcoef(m)
    # some features have 0 variance, so we must fix them
    np.fill_diagonal(cor, 1) 
    # cor = np.nan_to_num(cor)
    return cor