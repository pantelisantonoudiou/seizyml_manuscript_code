# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
##### ------------------------------------------------------------------- #####

metrics = ['percent_detected_seizures',  'f1', 'false_detected_rate']
hue_order = ['zscore', 'gaussian', 'quantile', 'minmax', ]
palette = ['#7b3294', '#c2a5cf', '#a6dba0', '#008837']
# =============================================================================
#                               Figure 7
# =============================================================================
if __name__ == '__main__':
    # 1 Mouse dataset
    data = pd.read_csv(os.path.join('data', 'trained_models', 'norm_comps', 'mouse', 'test_metrics.csv'))
    melted_data = pd.melt(data, id_vars=['kfold', 'norm_type', 'norm_strategy'], value_vars=metrics)
    g = sns.catplot(data=melted_data, x='norm_strategy', y='value', hue='norm_type', kind='bar',  hue_order=hue_order,
                    height=5, errorbar='se', col='variable',  col_wrap=3, sharey=False, palette=palette)

    # 2 MIT dataset - Inter-Subject
    data = pd.read_csv(os.path.join('data', 'trained_models', 'norm_comps', 'mit_chb', 'trained_models_inter','test_metrics_inter.csv'))
    data['percent_detected_seizures'] = 100*(data['detected_seizures']/data['total_seizures'])
    melted_data = pd.melt(data, id_vars=['group', 'norm_type', 'norm_strategy'], value_vars=metrics)
    g = sns.catplot(data=melted_data, x='norm_strategy', y='value', hue='norm_type', kind='bar',  hue_order=hue_order,
                    height=5, errorbar='se', col='variable',  col_wrap=3, sharey=False, palette=palette)

    plt.show()
