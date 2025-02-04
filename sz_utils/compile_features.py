# -*- coding: utf-8 -*-

### -------------- IMPORTS -------------- ###
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
### ------------------------------------- ###

def get_features(path, norm_func=None, norm=False,):
    """
      Constructs a dataset from .csv and .npz files in a specified directory.
    
      Parameters
      ----------
      path : str
          Path to the directory containing .csv and .npz files.
      norm_func : callable, optional
          Normalization function (e.g., a scaler class from `sklearn`) to normalize feature data.
          Only used if `norm` is True. Default is None.
      norm : bool, optional
          If True, applies normalization to the feature data using `norm_func`. Default is False.
    
      Returns
      -------
      x : numpy.ndarray
          Concatenated feature data from all .npz files.
      y_true : numpy.ndarray
          Concatenated target labels from all .csv files.
      """
     
    # get file paths
    csv_files = [x for x in os.listdir(path) if x[-4:] == '.csv']
    npz_files = [x.replace('.csv', '.npz') for x in csv_files]
    
    # iterate over files and construct dataset
    data_list = []
    y_true_list = []
    for npz_file, csv_file in tqdm(zip(npz_files, csv_files), total=len(csv_files)):
        y = pd.read_csv(os.path.join(path, csv_file), header=None)[0].values
        npz_file = np.load(os.path.join(path, npz_file))
        feature_data = npz_file['feature_data']
        if norm:
            feature_data = norm_func().fit_transform(feature_data)
        data_list.append(feature_data)
        y_true_list.append(y)
       
    # convert to proper format    
    x = np.concatenate(data_list, axis=0)
    y_true = np.concatenate(y_true_list)
        
    return x, y_true, npz_file['feature_labels']
    