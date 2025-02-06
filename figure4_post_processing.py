 # -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
##### ------------------------------------------------------------------- #####

if __name__ == '__main__':
    
    # load data
    data = pd.read_csv(os.path.join('data', 'trained_models', 'per_file', 'trained_models', 'post_processing.csv'))
    data['percent_seizures_detected'] = 100 * (data['detected_seizures'] / data['total_seizures'])

    # 1) plot anova metric
    metrics = ['f1', 'balanced_accuracy']
    colors= ['firebrick', 'skyblue']
    f, axs = plt.subplots(ncols=len(metrics), figsize=(20,10))
    for metric, color, ax in zip(metrics, colors, axs):
        formula = f'{metric} ~ post_processing + feature_set + model_name + + \
                post_processing:model_name + feature_set:model_name + post_processing:feature_set'
        model = ols(formula, data=data).fit()
        anova_results = sm.stats.anova_lm(model, typ=2)
        anova_results['eta_squared'] = anova_results['sum_sq'] / anova_results['sum_sq'].sum()
        print(anova_results)
        
        # Create a bar plot for the effect sizes
        effect_sizes = anova_results['eta_squared'][:-1].sort_values()
        effect_sizes.plot(kind='barh', color=color, ax=ax)
        ax.set_xlabel('Eta-Squared (Effect Size)')
        ax.set_title(f'Effect Sizes of Factors on {metric}')
        ax.set_yticks(range(len(effect_sizes)))
    plt.tight_layout()

    # 2) select 10 best features (middle ground as feature sets have small impact)
    plot_data = pd.DataFrame(data[data['feature_set'] == 'best_10'])
    
    # define plot parameters
    palette = ['#6D6E71', '#9FBCD3', '#AEC4B6']
    order = ['decision_tree', 'gaussian_nb', 'sgd', ]
    metrics = ['balanced_accuracy', 'recall', 'percent_seizures_detected', 
               'f1', 'precision', 'false_detected_rate', 'specificity']
    
    # 3) Melt dataframe and plot desired metrics
    melted_data = pd.melt(plot_data, id_vars=['post_processing', 'model_name', 'fold'], value_vars=metrics)
    g = sns.catplot(data=melted_data, y='post_processing', x='value',
                    height=10, errorbar='se', hue_order=order,
                      palette=palette, kind='point', col='variable',
                      col_wrap=3,  hue='model_name', sharex=False, )
    for ax in g.axes.flat:
        ax.grid(True, which='both', axis='both')
    plt.show()

  