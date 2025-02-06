# -*- coding: utf-8 -*-
### -------------- IMPORTS -------------- ###
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
### ------------------------------------- ###

def get_features(feature_path, norm_strategy, norm_func):
    """
     Normalize dataset features and collect labels for all subjects.
    
     This function processes data from a directory containing subject folders, 
     each with feature data and corresponding labels. It normalizes the feature data 
     based on the specified normalization strategy and function.
    
     Parameters:
     ----------
     feature_path : str
         Path to the root directory containing subject folders with `.csv` and `.npz` files.
     norm_strategy : str
         Normalization strategy to apply. Supported value: 'per_file'.
         If 'per_file', normalization is applied individually to each file's features.
     norm_func : callable
         A callable (e.g., sklearn Normalizer or StandardScaler) that performs 
         normalization. The callable must have `fit_transform` method.
    
     Returns:
     -------
     pd.DataFrame
         A DataFrame containing normalized feature data and corresponding labels.
         Each row represents one file, with the following columns:
         - 'subject': Name of the subject folder.
         - 'features': Normalized feature data (numpy array).
         - 'labels': Corresponding label values (numpy array).
    array-like
        With feature label names
    
     Example:
     --------
     ```python
     from sklearn.preprocessing import StandardScaler
    
     feature_path = "/path/to/data"
     data, feature_labels = norm_dataset(feature_path, norm_strategy='per_file', norm_func=StandardScaler)
     print(data.head())
     ```
     """
    folders = [f.name for f in os.scandir(feature_path) if f.is_dir()]
    feature_list = []
    for subject_folder in tqdm(folders, total=len(folders)):
        
        # get recordings per subject folder
        load_path = os.path.join(feature_path, subject_folder)
        files = os.listdir(load_path)
        csv_files = sorted([file for file in files if '.csv' in file])
        npz_files = sorted([file for file in files if '.npz' in file])
        
        for csv_file, npz_file in zip(csv_files, npz_files):
            # load features and labels
            y_true = pd.read_csv(os.path.join(load_path, csv_file))['0'].values
            npz_file = np.load(os.path.join(load_path, npz_file))
            feature_data = npz_file['feature_data']
            if norm_strategy == 'per_file':
                feature_data = norm_func().fit_transform(feature_data)
            feature_list.append({'subject':subject_folder, 'file':csv_file[:-4],'features':feature_data, 'labels':y_true})
    data = pd.DataFrame(feature_list)
    return data, npz_file['feature_labels']
