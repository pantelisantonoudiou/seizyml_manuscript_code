# -*- coding: utf-8 -*-

### -------------- IMPORTS -------------- ###
import os
import multiprocessing
import itertools
from joblib import dump
from uuid import uuid4
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sz_utils.feature_selection import select_features, get_feature_indices
from sz_utils.test_scores import get_scores_seizure_noclean
from sz_utils.compile_features_chbmit import get_features
from sz_utils.seizure_match import dual_threshold
### ------------------------------------- ###


                            ### User Settings ###
# ============================================================================= #
n_jobs = int(multiprocessing.cpu_count() * 0.7)
norm_types = {'zscore':StandardScaler, 'quantile':RobustScaler, 'minmax':MinMaxScaler, 'gaussian':PowerTransformer}
norm_strategies = ('per_file', 'all_files')
r_threshold = .9
feature_size = 10
outer_folds = 6
inner_folds = 4
# model settings
model_name = 'gaussian_nb'
models = {'gaussian_nb' : GaussianNB(),}
hyper_params = {'gaussian_nb': {'var_smoothing': np.logspace(-2,-8, num=7)},}
fit_metric = 'BALANCED_ACCURACY'
best_metric_rank = 'rank_test_' + fit_metric
metrics = {'AUC':'roc_auc', 'BALANCED_ACCURACY':'balanced_accuracy', 'PRECISION':'precision', 'RECALL':'recall', 'F1':'f1'}
grid_metrics = 'mean_test_'
# =============================================================================
# =============================================================================
    

def run_models():
    # paths
    model_path = os.path.join('data', 'saved_models', 'norm_comps', 'trained_models_chbmit')
    feature_path = os.path.join('data', 'features_chbmit')
    save_path = os.path.join(model_path, 'test_scores.csv')
    szr_amp_path = os.path.join(feature_path, 'szr_amp_per_file.csv')

    # create path if it does not exist
    os.makedirs(model_path, exist_ok=True)
    
    # exclude seizures with weak EEG profiles
    szr_amp_data = pd.read_csv(szr_amp_path)
    aver_data = szr_amp_data.groupby(['subject', 'file', 'seizure_id'])[['amp_change', 'seizure_duration']].median().reset_index()
    excluded = aver_data[(aver_data['amp_change']<1.5) | (aver_data['seizure_duration']<30)]
    excluded_files = excluded['file'].unique()
    
    print('\n --> Start training:')
    data_list = []
    for norm_type, norm_strategy in itertools.product(norm_types, norm_strategies):
        
        # get normalized features and exclude seizure files with non-robust EEG waveforms
        print(f'-> Getting normalized data {norm_strategy}, {norm_type}:')
        norm_func = norm_types[norm_type]
        df, feature_labels = get_features(feature_path, norm_strategy, norm_func)
        df = df[~df['file'].isin(excluded_files)]
        subjects = df['subject'].unique()
        
        # get subject groups based on outer folds and iterate over groups (based on outer folds)
        subject_lists = np.array(np.array_split(np.random.default_rng(seed=8).permutation(subjects), outer_folds))
        for group_idx in tqdm(range(outer_folds)):
            
            # get train and test datasets
            train_list = subject_lists[np.delete(np.arange(outer_folds) ,group_idx)].flatten()
            test_list = subject_lists[group_idx].flatten()
            x_train_val = np.concatenate(df[df['subject'].isin(train_list)]['features'].values)
            y_train_val = np.concatenate(df[df['subject'].isin(train_list)]['labels'].values)
            x_test = np.concatenate(df[df['subject'].isin(test_list)]['features'].values)
            y_test = np.concatenate(df[df['subject'].isin(test_list)]['labels'].values)
            
            # fill nans
            x_test = np.where(np.isnan(x_test), np.nanmedian(x_test, axis=0), x_test)
            x_train_val = np.where(np.isnan(x_train_val), np.nanmedian(x_train_val, axis=0), x_train_val)
            
            # normalize features
            if norm_strategy == 'all_files':
                x_test = norm_func().fit_transform(x_test)
                x_train_val = norm_func().fit_transform(x_train_val)
                
            # feature selection
            feature_space = select_features(x_train_val, y_train_val, feature_labels,
                                            r_threshold=r_threshold, feature_size=[feature_size],)
            selected_features = feature_space['best_'+str(feature_size)]
            sel_feature_idx = get_feature_indices(selected_features, feature_labels)
            
            # train and test models
            inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True)
            search = GridSearchCV(estimator=models[model_name],
                                param_grid=hyper_params[model_name],
                                scoring=metrics,
                                n_jobs=n_jobs,
                                cv=inner_cv,
                                verbose=0,
                                refit=fit_metric)
            search.fit(x_train_val[:, sel_feature_idx], y_train_val)

            # find best performing model and test
            best_model = search.best_estimator_
            y_pred = best_model.predict(x_test[:, sel_feature_idx])
            y_pred = dual_threshold(y_pred, t_high=.5, t_low=.2, win_size=6)
            scores = get_scores_seizure_noclean(y_test, y_pred)
            
            # save model with unique ID and append scores and other model info to dictionary
            best_model.features = selected_features
            uid = uuid4().hex
            dump(best_model, os.path.join(model_path, uid + '.joblib'))
            model_dict = {'id': uid, 'feature_id':'best_'+str(feature_size), 'model':model_name, 'group':group_idx,
                          'hyperparameters':search.best_params_, 'norm_type':norm_type, 'norm_strategy':norm_strategy}
            model_dict.update(scores)
            data_list.append(model_dict) 
            
    # convert list of dictionaries to dataframe and save
    data = pd.DataFrame(data_list)
    data.to_csv(save_path, index=False)

