# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
##### ------------------------------------------------------------------- #####

if __name__ == '__main__':
  
    
    # =============================================================================
    #                               SUPPLEMENTARY FIGURE 4
    # =============================================================================
    models = ['passive_aggressive', 'sgd']
    palette = ['#DBB391', '#AEC4B6']
    data = pd.read_csv(os.path.join('data', 'trained_models', 'per_file', 'trained_models', 'selected_models.csv'))
    plot_data = data.query("model_name == 'sgd'")
    plot_data = plot_data[plot_data['hyperparameters'].str.contains('hinge')]
    plot_data = pd.concat((plot_data, data.query("model_name == 'passive_aggressive'")))
    metrics = ['balanced_accuracy', 'recall', 'percent_detected',  'f1', 
               'precision', 'false_detected_rate', 'specificity']
    melted_data = pd.melt(plot_data, id_vars=['model_name', 'fold', 'feature_set'], value_vars=metrics)
    g = sns.catplot(data=melted_data, x='feature_set', y='value', hue='model_name', hue_order=models,
                    palette=palette, kind='bar', 
                    height=5, errorbar='se', col='variable',  col_wrap=3, sharey=False, sharex=False)
    for ax in g.axes.flat:
        ax.grid(True, which='both', axis='both')
    g.set_xticklabels(rotation=30)
    plt.show()

