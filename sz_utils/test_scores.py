# -*- coding: utf-8 -*-

#### ------------------------------ Imports ------------------------------ ####
import numpy as np
from sklearn import metrics
from sz_utils.seizure_match import get_szr_idx, match_szrs_idx
#### --------------------------------------------------------------------- ####

def basic_metrics(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp

def precision(tn, fp, fn, tp):
    precision = tp / (tp + fp)
    return precision

def recall(tn, fp, fn, tp):
    recall = tp / (tp + fn)
    return recall

def specificity(tn, fp, fn, tp):
    specificity = tn / (tn + fp)
    return specificity
    
def f1_score(tn, fp, fn, tp):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score

def balanced_accuracy(tn, fp, fn, tp):
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    return balanced_accuracy

model_metrics = {'balanced_accuracy': balanced_accuracy,
           'f1': f1_score,
           'precision': precision,
           'recall': recall,
           'specificity':specificity}

def get_scores(y_true, y_pred):
    """ get scores from metrics dict
    """
    
    scores = {}
    
    # get classification counts
    tn, fp, fn, tp = basic_metrics(y_true, y_pred)
    scores.update({'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp})
    
    # get basic model metrics
    for met, func in model_metrics.items():
        scores.update({met: func(tn, fp, fn, tp)})
        
    return scores

def get_scores_seizure_noclean(y_true, y_pred, win=5):
    """ get scores from metrics dict
    """
    
    # find bounds of true seizures (only seizures more than cutoff included)
    bounds_true = get_szr_idx(y_true)
    total_true = bounds_true.shape[0]
    rec_length = y_true.shape[0]*win/3600
    
    # get classification counts
    scores = {}
    tn, fp, fn, tp = basic_metrics(y_true, y_pred)
    scores.update({'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp})
    
    # get basic model metrics
    for met, func in model_metrics.items():
        scores.update({met: func(tn, fp, fn, tp)})
        
    # find percentage seizures detected and false positives
    bounds_pred = get_szr_idx(y_pred)
    ndetected = np.sum(match_szrs_idx(bounds_true, y_pred))
    false_detected_rate = (bounds_pred.shape[0] - ndetected)/rec_length
    
    # add seizure metrics
    scores.update({'detected_seizures':ndetected})
    scores.update({'total_seizures':total_true})
    scores.update({'false_detected_rate':false_detected_rate})
    scores.update({'data_length_hours':rec_length})
    scores.update({'percent_seizures_detected':100*ndetected/total_true})
    
    return scores