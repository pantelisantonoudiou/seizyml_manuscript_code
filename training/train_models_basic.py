# -*- coding: utf-8 -*-

#### ------------------------------ Imports ------------------------------ ####
import os
from uuid import uuid4
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from training.grid_search import grid_search_parallel
from sz_utils.feature_selection import select_features, get_feature_indices
from sz_utils.test_scores import get_scores_seizure_noclean
from sz_utils.compile_features import get_features
#### --------------------------------------------------------------------- ####

# helper functions to save and load the state as a DataFrame
def save_state(state_df, state_path):
    state_df.to_csv(state_path, index=False)

def load_state(state_path):
    if os.path.exists(state_path):
        return pd.read_csv(state_path)
    else:
        return pd.DataFrame()

def should_skip(state_df, fold, feature_type, feature_set, model_name):
    if len(state_df) == 0:
        return False
    return not state_df[(state_df['fold'] == fold) &
                        (state_df['feature_type'] == feature_type) &
                        (state_df['feature_set'] == feature_set) &
                        (state_df['model_name'] == model_name)].empty

def load_data(train_path, test_path, norm_func, norm_type):
    if norm_type == 'per_file':
        x_train_all, y_train_all, feature_labels_train = get_features(train_path, norm_func=norm_func, norm=True)
        x_test, y_test, feature_labels_test = get_features(test_path, norm_func=norm_func, norm=True)
    elif norm_type == 'all_file':
        x_train_all, y_train_all, feature_labels_train = get_features(train_path, norm_func=norm_func, norm=False)
        x_test, y_test, feature_labels_test = get_features(test_path, norm_func=norm_func, norm=False)
        x_train_all = norm_func().fit_transform(x_train_all)
        x_test = norm_func().fit_transform(x_test)
    else:
        raise Exception(f'--> Norm type did not match options (all_file/per_file) got {norm_type} instead.')

    if np.all(feature_labels_train == feature_labels_test):
        feature_labels = feature_labels_train
    else:
        print('Feature labels of train and test dataset do not match')
        return x_train_all, y_train_all, x_test, y_test, None
    return x_train_all, y_train_all, x_test, y_test, feature_labels

#### ----------------- Define models and hyperparameters ----------------- ####
# =============================================================================
models = {  
            'decision_tree': DecisionTreeClassifier(),
            'gaussian_nb': GaussianNB(),
            'passive_aggressive': PassiveAggressiveClassifier(),
            'sgd': SGDClassifier(),            
          }

hyper_params = {
                'decision_tree':{    
                                  'ccp_alpha': [0, 0.01,],
                                  'max_depth': [5, 10],
                                  'min_samples_leaf': [100, 1000, 10000],
                                  'max_features': [None],
                                  'class_weight': ['balanced']},
                
                'gaussian_nb': {
                                'var_smoothing': np.logspace(-2,-8, num=7)},

                'passive_aggressive' : {
                                        'C': [.1, 1, 10, 100],
                                        'class_weight': ['balanced']},
                
                'sgd' :{
                          'alpha': [0.001, 0.01,],
                          'max_iter': [1000,],
                          'learning_rate': ['adaptive'],
                          'penalty': ['l2', 'l1', None],
                          'early_stopping': [False],
                          'eta0': [0.001 ,0.01],
                          'tol' : [1e-3, 1e-4],
                          'class_weight': ['balanced'],
                          'loss': ['log_loss', 'hinge'],
                          'validation_fraction': [0.2],
                      },
                }
# =============================================================================
#### --------------------------------------------------------------------- ####

def run_models(norm_type='per_file'):
    # ---------------------------- Main settings ---------------------------- #
    # paths
    model_path = os.path.join('data', 'saved_models', norm_type, 'trained_models')
    state_path =  os.path.join(model_path, 'test_scores.csv')
    train_path = os.path.join('data', 'features_mouse', 'train')
    test_path = os.path.join('data', 'features_mouse', 'test')

    # create path if it does not exist
    os.makedirs(model_path, exist_ok=True)

    # parameters
    nsplits = 5
    random_state = 11
    feature_size = [5, 10, 15]
    r_threshold = .9
    nleast_corr = 5
    feature_types = ('local', 'relative', '_')
    n_models = len(feature_size)*2*nsplits*len(models)*len(feature_types)
    # ----------------------------------------------------------------------- #

    # load state and data
    state_df = load_state(state_path)
    x_train_all, y_train_all, x_test, y_test, feature_labels = load_data(train_path, test_path, norm_func=StandardScaler, norm_type=norm_type)
    print('--> Train and Test datasets were loaded.\n')

    # iterate over folds and feature selection first (ensures models see the same data)
    cv = StratifiedKFold(n_splits=nsplits, random_state=random_state, shuffle=True)
    cntr = 0
    for i, (train_index, val_index) in enumerate(cv.split(x_train_all, y_train_all)):
        for feature_type in feature_types:
            
            # select feature type before perfoming feature selection
            feature_idx = np.where(np.char.find(feature_labels, feature_type)>=0)[0]
            x_train = x_train_all[train_index, :]
            y_train = y_train_all[train_index]
            x_val = x_train_all[val_index, :]
            y_val = y_train_all[val_index]
            feature_space = select_features(x_train[:, feature_idx], y_train, feature_labels[feature_idx],
                                            r_threshold=r_threshold, feature_size=feature_size, 
                                            nleast_correlated=nleast_corr)

            # iterate through selected feature sets and model
            for feature_set, sel_features in feature_space.items():
                for model_name in models:
                    cntr+=1
                    # skip to the saved state
                    if should_skip(state_df, i, feature_type, feature_set, model_name):
                        print(f"-> Skipping: {cntr} out of {n_models}")
                        continue
                    else:
                        print(f'\n-> Performing grid search and testing model {cntr} out of {n_models}.')
                        print(f'unique y_train: {np.unique(y_train)}, unique y_val:{np.unique(y_val)}.')
                    
                    # perform the manual grid search
                    sel_feature_idx = get_feature_indices(sel_features, feature_labels) # remap back to full space
                    search = grid_search_parallel(model=models[model_name], hparams=hyper_params[model_name],
                                                x_train=x_train[:, sel_feature_idx], y_train=y_train, 
                                                x_val=x_val[:, sel_feature_idx],  y_val=y_val,
                                                scoring_function=balanced_accuracy_score)

                    # find best performing model and test
                    y_pred = search['best_model'].predict(x_test[:, sel_feature_idx])
                    scores = get_scores_seizure_noclean(y_test, y_pred)
                    
                    # save model outputs and state settings
                    model_id = uuid4().hex
                    model = search['best_model']
                    model.feature_labels = sel_features
                    dump(model, os.path.join(model_path, model_id + '.joblib'))
                    model_dict = {'id': model_id,  'fold':i, 'feature_type':feature_type,
                                    'feature_set':feature_set, 'model_name':model_name,
                                    'hyperparameters':str(search['best_params']), 'fit_metric':search['best_score']}
                    model_dict.update(scores)
                    state_df = pd.concat((state_df, pd.DataFrame(model_dict, index=[cntr])))
                    save_state(state_df, state_path)
