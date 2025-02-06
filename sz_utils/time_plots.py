# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load
import matplotlib.pyplot as plt
from sz_utils.feature_selection import get_feature_indices
##### ------------------------------------------------------------------- #####

def get_szbound_hist(true_szr_idx, pred_array, bins, time_bounds):
    """
    Generate histograms for true and predicted seizure boundaries.
    
    Parameters
    ----------
    true_szr_idx : numpy.ndarray
        2D array where each row contains [start, stop] indices of a true seizure.
    pred_array : numpy.ndarray
        1D boolean array representing predicted classification for seizures.
    bins : numpy.ndarray
        Time bins to consider for histogram creation.
    time_bounds : int
        Time bounds around the seizure event for prediction analysis, in units of bin_size.
        
    Returns
    -------
    data_true : numpy.ndarray
        1D array containing binned time points corresponding to true seizure boundaries, scaled by bin_size.
    data_pred : numpy.ndarray
        1D array containing binned time points where predictions are made around true seizures, corresponding to given bins.
        
    Example
    -------
    >>> get_szbound_hist(np.array([[10, 15], [20, 25]]), np.array([False, True, ...]), np.array([0, 1, ...]), 2)
    (array([0, 1, 2, ...]), array([0, 1, 2, ...]))
    """
    
    bin_size = bins[1] - bins[0] # get bin size
    data_true = np.array([])
    data_pred = np.array([])
    
    for seizure in true_szr_idx:
        
        # get seizure middle
        szr_mid = int(np.median(seizure))
        
        # get true seizure bounds
        szr_bounds = np.arange(seizure[0] - szr_mid, seizure[1] - szr_mid + 1, 1)
        data_true = np.concatenate((data_true, szr_bounds))

        # get predicted seizure bounds
        pred_around_seizure = pred_array[int(szr_mid - time_bounds/bin_size):int(szr_mid + time_bounds/bin_size)+1]
        data_pred = np.concatenate((data_pred, bins[pred_around_seizure]))
        
    return data_true * bin_size, data_pred

def get_time_bins(models_df, model_path, x_test, feature_labels, true_szr_idx, bins, time_bounds):
    """
    Generate a DataFrame containing time bins, model predictions, and ground truth labels.
    
    Parameters:
    models_df (DataFrame): Information about the best-performing models.
    model_path (str): Directory where trained model files are stored.
    feature_labels (array-like): Feature names.
    x_test (ndarray): Input data matrix with features for predictions.
    true_szr_idx (Index): Index indicating true seizure occurrences.
    bins : (numpy.ndarray): Time bins to consider for histogram creation.
    time_bounds : (int): Time bounds around the seizure event for prediction analysis, in units of bin_size.
    
    Returns:
    DataFrame: A DataFrame with time, model details, fold information, feature space ID, and classification.
    """
    
    df_list = []
    for idx, row in tqdm(models_df.iterrows(), total=len(models_df)):
        
        # load model, select feature and get binary predictions
        model = load(os.path.join(model_path, row['id'] +'.joblib'))
        sel_feature_idx = get_feature_indices(model.feature_labels, feature_labels)
        y_pred = model.predict(x_test[:,sel_feature_idx])>.5
        
        # get seizure bounds
        true_bounds, pred_bounds = get_szbound_hist(true_szr_idx, y_pred, bins, time_bounds)
        
        # append to dataframe
        temp = pd.DataFrame({'time': pred_bounds})
        temp['model_name'] = row['model_name']
        temp['fold'] = row['fold']
        temp['feature_set'] = row['feature_set']
        temp['classification'] = 'prediction'
        df_list.append(temp)
        
        temp = pd.DataFrame({'time': true_bounds})
        temp['model_name'] = row['model_name']
        temp['feature_set'] = row['feature_set']
        temp['classification'] = 'ground_truth'
        df_list.append(temp)

    df = pd.concat(df_list)
    return df

def plot_differences(df, models, bins):
    """
    Plots the differences in histogram counts between pairs of models.

    Parameters:
    df (DataFrame): The data containing the 'time' and 'model' columns.
    models (list): The list of model names to be compared.
    bins (array-like): The bin specification for the histograms.
   
    Returns:
    tuple: A tuple containing the figure and axes objects of the plot.
    """
    
    # Collect the counts and bin edges for each model
    counts = {}
    df_pred = df[df['classification'] == 'prediction']
    for model in models:
        data_model = df_pred[df_pred['model_name'] == model]['time']
        counts[model], _ = np.histogram(data_model, bins=bins)
    data_ground_truth = df[(df['model_name'] == model) & (df['classification'] == 'ground_truth')]['time']
    ground_truth_counts , _ = np.histogram(data_ground_truth, bins=bins) 

    # Compute the differences between counts of each pair of models
    differences = {}
    for model in models:
            differences[model] = ground_truth_counts - counts[model]

    # Plot the differences
    f_diff, axs = plt.subplots(nrows=len(models), ncols=1, sharey=True, sharex=True)
    for ax, (model, diff) in zip(axs.flatten(), differences.items()):
        colors = ['g' if d >= 0 else 'r' for d in diff] # green for positive, red for negative
        ax.bar(bins[:-1], diff, width=np.diff(bins), color=colors)
        ax.set_title(f"{model}")
        ax.tick_params(axis='both', which='both', direction='out', bottom=True, left=True) # Using ticks instead of grid
    f_diff.supxlabel('Time (seconds)')
    f_diff.supylabel('Difference in Counts')

def plot_model_differences(df, models, bins):
    """
    Plots the differences in histogram counts between pairs of models.

    Parameters:
    df (DataFrame): The data containing the 'time' and 'model' columns.
    models (list): The list of model names to be compared.
    bins (array-like): The bin specification for the histograms.
   
    Returns:
    tuple: A tuple containing the figure and axes objects of the plot.
    """
    
    # Collect the counts and bin edges for each model
    counts = {}
    for model in models:
        data_model = df[df['model_name'] == model]['time']
        counts[model], _ = np.histogram(data_model, bins=bins)

    # Compute the differences between counts of each pair of models
    differences = {}
    for i, model1 in enumerate(models):
        for model2 in models[i + 1:]:
            differences[(model1, model2)] = counts[model1] - counts[model2]

    # Plot the differences
    f_diff, axs_diff = plt.subplots(nrows=1, ncols=len(models), sharey=True, sharex=True)
    for ax, (pair, diff) in zip(axs_diff.flatten(), differences.items()):
        colors = ['g' if d >= 0 else 'r' for d in diff] # green for positive, red for negative
        ax.bar(bins[:-1], diff, width=np.diff(bins), color=colors)
        ax.set_title(f"{pair[0]} - {pair[1]}")
        ax.tick_params(axis='both', which='both', direction='out', bottom=True, left=True) # Using ticks instead of grid
    f_diff.supxlabel('Time (seconds)')
    f_diff.supylabel('Difference in Counts')

    return f_diff, axs_diff


def run_time_plots():

    # imports
    from sklearn.preprocessing import StandardScaler
    from sz_utils.seizure_match import get_szr_idx
    from training.train_models_basic import load_data

    # settings
    win = 5
    time_bounds = 200
    bins = np.arange(-time_bounds, time_bounds + 1, win)
    train_path = os.path.join('data', 'features_mouse', 'train')
    test_path = os.path.join('data', 'features_mouse', 'test')
    model_path = os.path.join('data', 'trained_models', 'per_file', 'trained_models')
    models_df = pd.read_csv(os.path.join(model_path, 'selected_models.csv'))
    save_path = os.path.join(model_path, 'time_predictions.csv')

    # laod data and get time predictions
    print('Loading Data:')
    _, _, x_test, y_test, feature_labels = load_data(train_path, test_path, norm_func=StandardScaler, norm_type='per_file')
    true_szr_idx = get_szr_idx(y_test)
    print('Calclulating Time Predictions:')
    df = get_time_bins(models_df, model_path, x_test, feature_labels, true_szr_idx, bins, time_bounds)
    df.to_csv(save_path, index=False)
