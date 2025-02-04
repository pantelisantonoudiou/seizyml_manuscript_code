# -*- coding: utf-8 -*-

#### ------------------------------ Imports ------------------------------ ####
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import contextlib
import joblib
#### --------------------------------------------------------------------- ####
n_jobs = int(multiprocessing.cpu_count() * 0.8)

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress
    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

def evaluate_model(model, model_params, x_train, y_train, x_val, y_val, scoring_function):
    """Helper function to fit the model and return the score."""
    model.set_params(**model_params)
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    score = scoring_function(y_val, y_val_pred)
    return score, model_params, model

def grid_search_serial(model, hparams, x_train, y_train, x_val, y_val, scoring_function):
    """
    Performs a manual grid search for the specified model using the provided
    training and validation data.

    Parameters:
    - model (estimator): The machine learning model to be tuned.
    - hparams (dict): Dictionary with parameters names (`str`) as keys and lists of
                      parameter settings to try as values.
    - x_train (array-like): The training feature matrix.
    - y_train (array-like): The training target vector.
    - x_val (array-like): The validation feature matrix.
    - y_val (array-like): The validation target vector.
    - scoring_function (function): The metric to optimize for.

    Returns:
    - output (dict): Containing best_model, best_params, and best_score.
    """

    
    # Iterate over all parameter combinations
    df_list = []
    param_grid = list(ParameterGrid(hparams))
    for model_params in tqdm(param_grid, desc='Grid search progress (serial)'):
        # Calculate model predictions
        score, output_params, trained_model = evaluate_model(clone(model), model_params, x_train, y_train, x_val, y_val, scoring_function)
        df_list.append({'model': trained_model, 'params': output_params, 'score': score})
    
    df = pd.DataFrame(df_list)
    output = df.loc[df['score'].idxmax()]
    output = dict(zip(['best_model', 'best_params', 'best_score'], output.values))
    return output

def grid_search_parallel(model, hparams, x_train, y_train, x_val, y_val, scoring_function, n_jobs=n_jobs):
    """
    Performs a manual grid search for the specified model using the provided
    training and validation data in parallel.

    Parameters:
    - model (estimator): The machine learning model to be tuned.
    - hparams (dict): Dictionary with parameters names (`str`) as keys and lists of
                      parameter settings to try as values.
    - x_train (array-like): The training feature matrix.
    - y_train (array-like): The training target vector.
    - x_val (array-like): The validation feature matrix.
    - y_val (array-like): The validation target vector.
    - scoring_function (function): The metric to optimize for.
    - n_jobs (int): The number of jobs to run in parallel.

    Returns:
    - output (dict): Containing best_model, best_params, and best_score.
    """
    
    param_grid = list(ParameterGrid(hparams))
    with tqdm_joblib(tqdm(total=len(param_grid), desc="Grid Search Progress (parallel)")) as progress_bar:  # NOQA
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_model)(clone(model), model_params, x_train, y_train, x_val, y_val, scoring_function)
            for model_params in param_grid)
                        
    # Find the best result
    best_score, best_params, best_model = max(results, key=lambda x: x[0])
    output = {'best_model': best_model, 'best_params': best_params, 'best_score': best_score}
    return output
