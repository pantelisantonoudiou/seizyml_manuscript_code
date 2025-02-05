# -*- coding: utf-8 -*-
### -------------- IMPORTS -------------- ###
import os
import itertools
from joblib import dump
from uuid import uuid4
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from training.grid_search import grid_search_parallel
from sz_utils.feature_selection import select_features, get_feature_indices
from sz_utils.test_scores import get_scores_seizure_noclean
from sz_utils.seizure_match import dual_threshold
from training.train_models_basic import load_data
### ------------------------------------- ###


                        ### User Settings ###
# ============================================================================= #
norm_methods = {'zscore':StandardScaler, 'quantile':RobustScaler, 'minmax':MinMaxScaler, 'gaussian':PowerTransformer}
norm_types = ('per_file', 'all_files')
nfolds = 5
r_threshold = .9
feature_size = 10
random_state = 11
model_name = 'gaussian_nb'
models = {'gaussian_nb' : GaussianNB(),}
hyper_params = {'gaussian_nb': {'var_smoothing': np.logspace(-2,-8, num=7)},}
# =============================================================================
# =============================================================================

def run_models():

    # paths
    model_path = os.path.join('data', 'saved_models', 'norm_comps', 'trained_models_mouse')
    train_path = os.path.join('data', 'features_mouse', 'train')
    test_path = os.path.join('data', 'features_mouse', 'test')
    save_path = os.path.join(model_path, 'test_scores.csv')

    # create path if it does not exist
    os.makedirs(model_path, exist_ok=True)

    # get feature labels and iterate over normalization types
    print('\n --> Start training:')
    data_list = []
    for norm_method, norm_type in itertools.product(norm_methods, norm_types):
        
        # get normalized features depending on strategy and norm function
        print(f'-> Training {norm_method}, {norm_type}')
        norm_func = norm_methods[norm_method]
        x_train_val, y_train_val, x_test, y_test, feature_labels = load_data(train_path, test_path, norm_func=norm_func, norm_type=norm_type)
           
        # iterate over folds and feature selection first (ensures models see the same data)
        cv = StratifiedKFold(n_splits=nfolds, random_state=random_state, shuffle=True)
        for i, (train_index, val_index) in enumerate(cv.split(x_train_val, y_train_val)):
            print(f'--> Fold: {i+1} out of {nfolds}.')
  
            # split datasets and perform feature selection
            x_train = x_train_val[train_index, :]
            y_train = y_train_val[train_index]
            x_val = x_train_val[val_index, :]
            y_val = y_train_val[val_index]
            feature_space = select_features(x_train, y_train, feature_labels,
                                            r_threshold=r_threshold, feature_size=[feature_size], 
                                            nleast_correlated=0)
            selected_features = feature_space['best_'+str(feature_size)]
            sel_feature_idx = get_feature_indices(selected_features, feature_labels)
            
            # perform the manual grid search
            sel_feature_idx = get_feature_indices(selected_features, feature_labels) # remap back to full space
            search = grid_search_parallel(model=models[model_name], hparams=hyper_params[model_name],
                                        x_train=x_train[:, sel_feature_idx], y_train=y_train, 
                                        x_val=x_val[:, sel_feature_idx],  y_val=y_val,
                                        scoring_function=balanced_accuracy_score)

            # find best performing model and test
            y_pred = search['best_model'].predict(x_test[:, sel_feature_idx])
            y_pred = dual_threshold(y_pred, t_high=.5, t_low=.2, win_size=6)
            scores = get_scores_seizure_noclean(y_test, y_pred)

            # save model with unique ID and append scores and other model info to dictionary
            model = search['best_model']
            model.features = selected_features
            uid = uuid4().hex
            dump(model, os.path.join(model_path, uid + '.joblib'))
            model_dict = {'id': uid, 'feature_id':'best_'+str(feature_size), 'model':model_name, 'kfold':i,
                          'hyperparameters':str(search['best_params']), 'norm_type':norm_type, 'norm_method':norm_method}
            model_dict.update(scores)
            data_list.append(model_dict)
            
    # convert list of dictionaries to dataframe and save
    data = pd.DataFrame(data_list)
    data.to_csv(save_path, index=False)
