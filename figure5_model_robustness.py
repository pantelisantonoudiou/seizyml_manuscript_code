# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
##### ------------------------------------------------------------------- #####

def set_log_scale_x_one(g, ticks=[1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]):
    """
    Set a logarithmic scale for the x-axis of the provided seaborn plot.
    
    Parameters:
        g (matplotlib.axes._subplots.AxesSubplot): A single matplotlib subplot.
        ticks (list): List of tick positions for the x-axis. Default is [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0].
    """
    # Apply the ticks to the plot
    ax = g.axes
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('{:.1f}'.format(x))))
##### ------------------------------------------------------------------- #####

# =============================================================================
#                               Figure 5
# =============================================================================
if __name__ == '__main__':
    
    # =============================================================================
    #                       A) Vary Training Size 
    # =============================================================================
    # load data and select post processed data and best_10 feature_set
    palette = ['#6D6E71', '#9FBCD3','#AEC4B6'] 
    data = pd.read_csv(os.path.join('data', 'trained_models', 'per_file','trained_models_var_size', 'test_scores_pp.csv'))
    data['percent_dataset_size'] = data['dataset_fraction']*100
    plot_data = data.query("feature_set == 'best_10' and post_processing == True")
    
    # plot
    metrics = ['percent_seizures_detected', 'false_detected_rate', 'balanced_accuracy', 'f1', 'precision', 'recall']
    f, axs = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(12,5))
    for metric,ax in zip(metrics, axs.flatten()):
        g = sns.lineplot(data=plot_data, x='percent_dataset_size', y=metric, hue='model_name', errorbar='se', 
                      palette=palette, marker='o', markersize=15, ax=ax)
        set_log_scale_x_one(g, ticks=data['percent_dataset_size'].unique().tolist())
        ax.legend().set_visible(False)
        ax.grid(linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')  
    plt.tight_layout()
    
    # =============================================================================
    #                       B) Permute labels
    # =============================================================================
    # load data and select post processed data and best_10 feature_set
    palette = ['#6D6E71', '#9FBCD3','#AEC4B6'] 
    data = pd.read_csv(os.path.join('data', 'trained_models', 'per_file','trained_models_permute_labels', 'test_scores_pp.csv'))
    plot_data = pd.DataFrame(data.query("feature_set == 'best_10' and post_processing == True"))
    plot_data['percentage_permuted'] = plot_data['percentage_permuted'] + 0.1
    
    # plot
    metrics = ['percent_seizures_detected', 'false_detected_rate', 'balanced_accuracy', 'f1', 'precision', 'recall']
    f, axs = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(12,5))
    for metric,ax in zip(metrics, axs.flatten()):
        g = sns.lineplot(data=plot_data, x='percentage_permuted', y=metric, hue='model_name', errorbar='se', 
                      palette=palette, marker='o', markersize=15, ax=ax)
        set_log_scale_x_one(g, ticks=plot_data['percentage_permuted'].unique().tolist())
        ax.legend().set_visible(False)
        ax.grid(linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
    plt.tight_layout()

    # =============================================================================
    #                       C) Permute labels abnd Vary Training Size
    # =============================================================================
    # load data and select post processed data and best_10 feature_set
    palette = ['#6D6E71', '#9FBCD3','#AEC4B6'] 
    data = pd.read_csv(os.path.join('data', 'trained_models', 'per_file','trained_small_models_permute_labels', 'test_scores_pp.csv'))
    data['percent_dataset_size'] = data['training_fraction']*100
    data['percent_random_labels'] = (data['fraction_permuted_labels']*100 )
    plot_data = data.query("feature_set == 'best_10' and post_processing == True and percent_dataset_size == 1.0")
    
    # plot
    metrics = ['percent_seizures_detected', 'false_detected_rate', 'balanced_accuracy',]# 'f1', 'balanced_accuracy']
    f, axs = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(12,3))
    for metric,ax in zip(metrics, axs.flatten()):
        g = sns.lineplot(data=plot_data, x='percent_random_labels', y=metric, hue='model_name', errorbar='se', 
                      palette=palette, marker='o', markersize=12, ax=ax)
        # set_log_scale_x_one(g, ticks=plot_data['percent_random_labels'].unique().tolist())
        ax.legend().set_visible(False)
        ax.grid(linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()
    
    