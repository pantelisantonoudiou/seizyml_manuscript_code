# -*- coding: utf-8 -*-
### -------------- IMPORTS -------------- ###
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
### ------------------------------------- ###
    
if __name__ == '__main__':
    
    # =============================================================================
    #                                 Supplementary Figure 5
    # =============================================================================
    
    # settings
    win = 5
    time_bounds = 200
    bins = np.arange(-time_bounds, time_bounds + 1, win)
    
    # load data and plot
    time_data = pd.read_csv(os.path.join('data', 'trained_models', 'per_file', 'trained_models', 'time_predictions_gnb.csv'))
    score_data = pd.read_csv(os.path.join('data', 'trained_models', 'per_file', 'trained_models', 'gnb_scores_one_feature.csv'))
    plt.figure()
    sns.barplot(data=score_data, x='model', y='f1', errorbar='se')
    plt.figure()
    sns.histplot(data=time_data, x='time', bins=bins, stat='count', hue='classification', linewidth=1,)
    plt.show()