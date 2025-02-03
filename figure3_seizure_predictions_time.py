
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sz_utils.time_plots import plot_differences
sns.set(font_scale=2)
sns.set(style="whitegrid")
##### ------------------------------------------------------------------- #####

if __name__ == '__main__':
    
    # create time bins
    win = 5
    time_bounds = 200
    bins = np.arange(-time_bounds, time_bounds + 1, win)
    
# =============================================================================
#                               FIGURE 4
# =============================================================================
    
    # load data
    df = pd.read_csv(os.path.join('data', 'trained_models', 'per_file', 'trained_models', 'time_predictions.csv'))

    # plot seizure prediction over time
    models = ['decision_tree', 'gaussian_nb', 'passive_aggressive', 'sgd']
    ground_truth_color = '#b53b33'
    palette = ['#6D6E71', '#9FBCD3', '#DBB391', '#AEC4B6']
    
    # plot model vs ground truth
    f, axs = plt.subplots(nrows=len(models), ncols=1, sharey=True, sharex=True)
    axs = axs.flatten()
    
    # Loop through each model, plot predictions and overlay ground truth
    for model, color, ax in zip(models, palette, axs):
        # Plot the model's prediction
        sns.histplot(data=df[df['model_name']==model], x='time', bins=bins, stat='count',
                     hue='classification', hue_order=['prediction'], linewidth=1,legend=False,
                     palette=[color], ax=ax)
    
        # Overlay the ground truth as an outline on the same subplot
        sns.histplot(data=df[df['model_name']=='sgd'], x='time', bins=bins, stat='count',
                     hue='classification', hue_order=['ground_truth'], linewidth=2,legend=False,
                     palette=[ground_truth_color], ax=ax, fill=False, element='step')
    
        # Adjust ticks and labels
        ax.tick_params(axis='both', which='both', direction='out', bottom=True, left=True)
        ax.set_title(model)
    
    # plot differences
    plot_differences(df, models, bins)
    
    # =============================================================================
    #                               SUPPLEMENTARY FIGURE 3
    # =============================================================================
    
    # all feature sets
    sns.displot(data=df, x='time', bins=bins, stat='count', kind='hist', col='model_name', row='feature_set',
                hue='classification', hue_order=['ground_truth', 'prediction'],linewidth=1, palette=['#9FBCD3','#DBB391', ])

