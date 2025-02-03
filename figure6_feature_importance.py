 # -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
##### ------------------------------------------------------------------- #####

def get_importances(model_df, model_path, conds) :
    """
    Retrieve feature importances from various models and compile them into a DataFrame.

    Parameters
    ----------
    model_df : pd.Dataframe, containing meta-information about each model, including the model type and feature space identifier.

    Returns
    -------
    df : pd.Dataframe, A DataFrame containing feature importances. Each row corresponds to a feature for a given model.
        The DataFrame has columns for 'features', 'importance', and 'model'.
    """
    df_list = []
    for idx,row in tqdm(model_df.iterrows(), total=len(model_df)):
        
        # select features and load model
        model = load(os.path.join(model_path, row['id'] +'.joblib'))
      
        # get feature importances
        if row['model_name'] == 'decision_tree':
            importances = model.feature_importances_
            
        elif row['model_name'] == 'sgd':
            importances = abs(model.coef_[0])
            
        elif row['model_name'] == 'gaussian_nb':
            importances =  np.abs(model.theta_[0] - model.theta_[1]) / (np.sqrt(model.var_[0]) + np.sqrt(model.var_[1]))
            
        importances = importances/np.sum(importances)
        temp_df = pd.DataFrame([row.values], columns=row.index)
        temp_df = pd.concat([temp_df]*len(model.feature_labels))
        temp_df['features'] = model.feature_labels
        temp_df['importance'] = importances
        temp_df[conds] = row[conds]
        df_list.append(temp_df)
    df = pd.concat(df_list)
    return df

if __name__ == '__main__':
    
    # =============================================================================
    #                                 Figure 6
    # =============================================================================
    
    ### models trained on 1% no label permutation
    model_path = os.path.join('data', 'trained_models', 'per_file', 'trained_small_models_permute_labels')
    data = pd.read_csv(os.path.join(model_path,  'test_scores.csv'))
    data['percent_dataset_size'] = data['training_fraction']*100
    data['percent_random_labels'] = (data['fraction_permuted_labels']*100 )
    models_df = data.query("feature_set == 'best_10' and percent_dataset_size == 1.0 and percent_random_labels==0") #
    models_df = models_df[['model_name', 'id', 'fold', 'feature_set',]]
    models_df = get_importances(models_df, model_path, conds=['model_name'])
    plot_data = models_df.groupby(['model_name', 'features']).sum(numeric_only=True).reset_index()
    plot_data['norm_importance'] = plot_data['importance']/plot_data.groupby(['model_name'])['importance'].transform('sum')
    plot_data = plot_data.sort_values(['model_name', 'norm_importance'], ascending=[True, False])
    g = sns.catplot(data=plot_data, y='features', x='norm_importance', kind='bar', height=10, row='model_name', sharex=True)
    
    ### models trained on 1% of data with permute labels
    data = pd.read_csv(os.path.join(model_path, 'test_scores.csv'))
    data['percent_dataset_size'] = data['training_fraction']*100
    data['percent_random_labels'] = (data['fraction_permuted_labels']*100 )
    models_df = data.query("feature_set == 'best_10' and percent_dataset_size == 1.0") #
    models_df = models_df[['model_name', 'id', 'fold', 'feature_set', 'percent_random_labels']]
    models_df = get_importances(models_df, model_path, conds=['model_name','percent_random_labels', 'fold'])
    plot_data = models_df.groupby(['model_name', 'percent_random_labels', 'features']).sum(numeric_only=True).reset_index()
    plot_data['norm_importance'] = plot_data['importance']/plot_data.groupby(['model_name', 'percent_random_labels'])['importance'].transform('sum')
    
    # Pivot data to have percent_random_labels as columns and features as rows, for each model
    heatmap_data = {}
    linemap_data = {}
    reference = data['percent_random_labels'].min()
    for model in plot_data['model_name'].unique():
        model_data = plot_data[plot_data['model_name'] == model]
        pivot_data = model_data.pivot(index='features', columns='percent_random_labels', values='norm_importance')
        # pivot_data = pivot_data.dropna(axis=0, how='all')
        baseline = pivot_data[reference].fillna(value=0)
        diff_data = pivot_data.subtract(baseline, axis=0)
        diff_data = diff_data.fillna(value=0)
        heatmap_data[model] = diff_data
        linemap_data[model] = diff_data.abs().sum(axis=0)
    
    # Plot heatmap for each model
    for model, df in heatmap_data.items():
        plt.figure(figsize=(12, 8))
        max_val = int(np.max([np.abs(df.min())] +[np.abs(df.max())])*100)/100
        sns.heatmap(df, cmap="vlag_r", annot=False, fmt=".2f", cbar_kws={'label': 'Importance Difference'},
                    vmin=-max_val, vmax=max_val, linewidths=.1, xticklabels=True, yticklabels=True)
        plt.title(f"Importance Difference Heatmap for {model}")
        plt.xlabel("Percent Random Labels")
        plt.ylabel("Features")
        plt.tight_layout()

    