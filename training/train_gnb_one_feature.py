# -*- coding: utf-8 -*-
### -------------- IMPORTS -------------- ###
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from training.grid_search import grid_search_parallel
from sz_utils.test_scores import get_scores_seizure_noclean
from sz_utils.seizure_match import dual_threshold, get_szr_idx
from training.train_models_basic import load_data
from sz_utils.time_plots import get_szbound_hist
### ------------------------------------- ###


                        ### User Settings ###
# ============================================================================= #
nfolds = 5
r_threshold = .9
random_state = 11
feature = 'local_line_length_vhpc'
model_name = 'gaussian_nb'
models = {'gaussian_nb' : GaussianNB(),}
hyper_params = {'gaussian_nb': {'var_smoothing': np.logspace(-2,-8, num=7)},}

# create time bins
fraction = 0.1
win = 5
time_bounds = 200
bins = np.arange(-time_bounds, time_bounds + 1, win)
# =============================================================================
# =============================================================================

def run_models():

    # paths
    train_path = os.path.join('data', 'features_mouse', 'train')
    test_path = os.path.join('data', 'features_mouse', 'test')
    save_dir = os.path.join('data', 'plot_data')
    save_path_df = os.path.join(save_dir, 'time_predictions_gnb.csv')
    save_path_scores = os.path.join(save_dir, 'gnb_scores_one_feature.csv')

    # create path if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # load dataset and get 1 % fraction
    x_train_all, y_train_all, x_test, y_test, feature_labels = load_data(train_path, test_path, norm_func=StandardScaler, norm_type='per_file')
    x_train_val, _, y_train_val, _ = train_test_split(x_train_all, y_train_all, train_size=fraction, 
                                                stratify=y_train_all, random_state=random_state)
    
    # iterate over folds and feature selection first (ensures models see the same data)
    data_list = []
    score_list = []
    cv = StratifiedKFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    for i, (train_index, val_index) in enumerate(cv.split(x_train_val, y_train_val)):
        print(f'--> Fold: {i+1} out of {nfolds}.')
  
        # split datasets and perform feature selection
        x_train = x_train_val[train_index, :]
        y_train = y_train_val[train_index]
        x_val = x_train_val[val_index, :]
        y_val = y_train_val[val_index]

        # perform the manual grid search
        sel_feature_idx = np.where(feature_labels == feature)[0]
        search = grid_search_parallel(model=models[model_name], hparams=hyper_params[model_name],
                                    x_train=x_train[:, sel_feature_idx], y_train=y_train, 
                                    x_val=x_val[:, sel_feature_idx],  y_val=y_val,
                                    scoring_function=balanced_accuracy_score)
        
        # find best performing model and test
        y_pred = search['best_model'].predict(x_test[:, sel_feature_idx])>.5
        scores = get_scores_seizure_noclean(y_test, dual_threshold(y_pred))
        scores.update({'fold':i, 'model':model_name, 'feature':feature})
        score_list.append(scores)
        
        # get time predictions
        true_szr_idx = get_szr_idx(y_test)
        true_bounds, pred_bounds = get_szbound_hist(true_szr_idx, y_pred, bins, time_bounds)
        temp = pd.DataFrame({'time': pred_bounds})
        temp['classification'] = 'prediction'
        data_list.append(temp)
        temp = pd.DataFrame({'time': true_bounds})
        temp['classification'] = 'ground_truth'
        data_list.append(temp)
        
    # concatenate data
    data = pd.concat(data_list).reset_index()
    scoredf = pd.DataFrame(score_list).to_csv(save_path_scores, index=False)
    data.to_csv(save_path_df, index=False)
