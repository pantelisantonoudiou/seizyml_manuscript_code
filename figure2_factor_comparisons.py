# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
##### ------------------------------------------------------------------- #####

if __name__ == '__main__':
    
    # settings
    export_folder = 'stats'
    palette = ['#6D6E71', '#9FBCD3', '#DBB391', '#AEC4B6']
    order = ['decision_tree', 'gaussian_nb', 'passive_aggressive', 'sgd', ]
    
    # load data
    df1 = pd.read_csv(os.path.join('data', 'trained_models', 'all_files', 'trained_models', 'test_scores.csv'))
    df1['norm_type'] = 'all_files'
    df2 = pd.read_csv(os.path.join('data', 'trained_models', 'per_file', 'trained_models', 'test_scores.csv'))
    df2['norm_type'] = 'per_file'
    data = pd.concat((df1, df2), axis=0).reset_index(drop=True)
    data.loc[data['feature_type']=='_', 'feature_type'] = 'local+relative'
    data['percent_detected'] = 100*(data['detected_seizures']/data['total_seizures'])
    data['feature_type'] = data['feature_type'].astype('category')
    data['feature_set'] = data['feature_set'].astype('category') 
    data['model_name'] = data['model_name'].astype('category')
    data['norm_type'] = data['norm_type'].astype('category')
    
    # =============================================================================
    #                               FIGURE 2
    # =============================================================================
    
    # 1) Perform ANOVA and plot main effect size
    metrics = ['f1', 'balanced_accuracy']
    colors= ['firebrick', 'skyblue']
    f, axs = plt.subplots(ncols=len(metrics))
    for metric, color, ax in zip(metrics, colors, axs):
        formula = f'{metric} ~ feature_type + feature_set + model_name + norm_type + \
                feature_type:model_name + feature_set:model_name + feature_type:feature_set + \
                feature_type:norm_type + model_name:norm_type + feature_set:norm_type'
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

    # 2) Compare norm types
    g = sns.catplot(data=data, y='model_name', x='balanced_accuracy', hue='norm_type', kind='strip', order=order,)
    g = sns.catplot(data=data, y='model_name', x='f1', hue='norm_type', kind='strip', order=order,)
    
    # 3) Feature Type
    data = pd.DataFrame(data[data['norm_type'] =='per_file'].reset_index(drop=True)) # select normalization per file
    g = sns.catplot(data=data, y='model_name', hue='feature_type', x='f1', kind='strip', order=order)
    g = sns.catplot(data=data, y='model_name', hue='feature_type', x='balanced_accuracy', kind='strip', order=order)
    
    # 4) Feature Set
    data = pd.DataFrame(data[data['feature_type'] =='local'].reset_index(drop=True)) # select only local features
    hue_order = ['best_5', 'best_10', 'best_15', 'best_5_and_leastcorr', 'best_10_and_leastcorr', 'best_15_and_leastcorr']
    g = sns.catplot(data=data, x='model_name', hue='feature_set', y='balanced_accuracy', kind='bar', errorbar='se', 
                    order=order, hue_order=hue_order, palette='rocket')
    g = sns.catplot(data=data, x='model_name', hue='feature_set', y='f1', kind='bar', errorbar='se',
                    order=order, hue_order=hue_order, palette='rocket')
    data.to_csv(os.path.join('data', 'trained_models', 'per_file', 'trained_models', 'selected_models.csv'), index=False)
    plt.show()
 