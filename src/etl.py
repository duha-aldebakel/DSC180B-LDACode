import os
import pandas as pd
import bz2
import pickle

def load_data(data_dir, file_name, **kwargs):
    path = os.path.join(data_dir, file_name)

    print('Loading data... ', end = "")
    with bz2.BZ2File(path, 'rb') as f:  #Use datacompression BZ2
        data = pickle.load(f)
    print('Done.')

    return data
