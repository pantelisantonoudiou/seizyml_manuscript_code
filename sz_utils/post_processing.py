##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
from scipy import ndimage
import pandas as pd
from tqdm import tqdm
from joblib import load
from joblib import Parallel, delayed
from sz_utils.feature_selection import get_feature_indices
from sz_utils.test_scores import get_scores_seizure_noclean
from training.train_models_basic import load_data
from sklearn.preprocessing import StandardScaler
##### ------------------------------------------------------------------- #####

def hysteresis_vectorized(mean_pred, T_high, T_low):
    """
    Apply hysteresis thresholding to detect events in a signal.

    Parameters:
    mean_pred (np.array): The smoothed signal (e.g., moving average)
    T_high (float): The high threshold for triggering an event
    T_low (float): The low threshold for ending an event

    Returns:
    np.array: Binary event vector after hysteresis thresholding
    """

    seeds = mean_pred >= T_high
    mask = mean_pred > T_low
    hysteresis_output = ndimage.binary_propagation(seeds, mask=mask)
    return hysteresis_output.astype(int)

def clean_predictions(vector, dilation=0, erosion=0, operation_type='dilation_erosion'):
    """
    Applies morphological operations to a 1D binary vector.

    This function processes the input binary vector by applying morphological operations in a specified order,
    determined by the `operation_type` parameter. The vector is padded with zeros at both ends to prevent
    boundary effects during the operations.

    Parameters:
        vector (numpy.ndarray): Input 1D binary vector.
        dilation (int): Size of the dilation structuring element minus one (default is 0).
        erosion (int): Size of the erosion structuring element (default is 0).
        operation_type (str): Order of operations to perform; either 'dilation_erosion' (default) or 'erosion_dilation'.

    Returns:
        numpy.ndarray: The processed vector after applying the specified morphological operations, with padding removed.

    """
    padded_vector = np.concatenate(([0,0,0], vector, [0,0,0]))
    if operation_type == 'dilation_erosion':
        operation1 = ndimage.binary_closing(padded_vector, structure=np.ones(dilation+1)).astype(int)
        operation2 = ndimage.binary_opening(operation1, structure=np.ones(erosion+1)).astype(int)
    elif operation_type == 'erosion_dilation':
        operation1 = ndimage.binary_opening(padded_vector, structure=np.ones(erosion+1)).astype(int)
        operation2 = ndimage.binary_closing(operation1, structure=np.ones(dilation+1)).astype(int)
    return operation2[3:-3]

def get_post_processing(raw_pred, y_true, param_dict):
    """
    Post-processes raw predictions based on specified parameters for either morphological 
    operations (dilation and erosion) or rolling window mean followed by hysteresis thresholding.
    
    Parameters:
    ----------
    raw_pred : array-like, The raw predictions to be post-processed.
    y_true :  array-, Ground truth labels.
    param_dict : dict, Dictionary containing the parameters for post-processing.
        
    
    Returns:
    -------
    scores : dict
        The performance metrics (e.g., accuracy, precision, recall) computed on the clean predictions.
    
    post_processing : str
        A string summarizing the post-processing steps applied (useful for tracking or logging).
    """
    
    if 'dilation' in param_dict.keys():
        clean_pred = clean_predictions(raw_pred, dilation=param_dict['dilation'], 
                                        erosion=param_dict['erosion'], 
                                        operation_type=param_dict['operation_type']
                                        )
        post_processing = f"{param_dict['operation_type']}_dilation={param_dict['dilation']}_"
        post_processing += f"erosion={param_dict['erosion']}"
        
    elif 'rolling_window_size' in param_dict.keys():
        win_size = param_dict['rolling_window_size']
        mean_pred = np.convolve(raw_pred, np.ones(win_size)/win_size, mode='same')
        clean_pred = hysteresis_vectorized(mean_pred, 
                                            param_dict['high_threshold'],
                                            param_dict['low_threshold'])
        post_processing = f"rolling_mean_{param_dict['rolling_window_size']}_{param_dict['high_threshold']}_"
        post_processing += f"{param_dict['low_threshold']}"
        
    else:
        print('no matching keyword was found in dictionary')
        
    scores = get_scores_seizure_noclean(y_true, clean_pred)
    return scores, post_processing


def run_processing():

    # paths
    model_path = os.path.join('data', 'trained_models', 'per_file', 'trained_models')
    train_path = os.path.join('data', 'features_mouse', 'train')
    test_path = os.path.join('data', 'features_mouse', 'test')
    save_path = os.path.join(model_path, 'post_processing.csv')
    _, _, x_test, y_true, feature_labels = load_data(train_path, test_path, norm_func=StandardScaler, norm_type='per_file')

    # get train models and dataframe
    trained_models = pd.read_csv(os.path.join(model_path, 'selected_models.csv'))
    trained_models = trained_models[trained_models['model_name'] != 'passive_aggressive']
    
    # parameters
    parameters = ({'dilation':2, 'erosion':1, 'operation_type':'dilation_erosion'},
                  {'dilation':2, 'erosion':2, 'operation_type':'dilation_erosion'},
                  {'dilation':3, 'erosion':3, 'operation_type':'dilation_erosion'},
                  {'erosion':1, 'dilation':2, 'operation_type':'erosion_dilation'},
                  {'erosion':1, 'dilation':3, 'operation_type':'erosion_dilation'},
                  {'erosion':2, 'dilation':4, 'operation_type':'erosion_dilation'},
                  {'rolling_window_size':6, 'high_threshold':0.5, 'low_threshold':0.2},
                  {'rolling_window_size':8, 'high_threshold':0.5, 'low_threshold':0.2},
                  {'rolling_window_size':10, 'high_threshold':0.5, 'low_threshold':0.2},
                  )
    
    # get model predictions
    data_list = []
    for i, row in tqdm(trained_models.iterrows(), total=len(trained_models)):
        
        # load model and get predictions
        model = load(os.path.join(model_path, row['id'] +'.joblib'))
        sel_feature_idx = get_feature_indices(model.feature_labels, feature_labels)
        raw_pred = model.predict(x_test[:,sel_feature_idx])   

        # get model paramtetrs and raw scores
        model_dict = dict(row[['id', 'fold', 'feature_type', 'feature_set', 'model_name',
                'hyperparameters', 'fit_metric', 'norm_type',]])
        scores = get_scores_seizure_noclean(y_true, raw_pred)
        data_list.append({**model_dict, 'post_processing':'none', **scores})
        
        # add scores from post processing
        output = Parallel(n_jobs=10, backend='loky')(delayed(get_post_processing)(raw_pred, y_true, param_dict) for param_dict in parameters)
        for scores, post_processing in output:
            data_list.append({**model_dict, 'post_processing':post_processing, **scores})
    data = pd.concat([pd.DataFrame([x]) for x in data_list], ignore_index=True)
    data['percent_seizures_detected'] = 100 * (data['detected_seizures'] / data['total_seizures'])
    data.to_csv(save_path, index=False)
