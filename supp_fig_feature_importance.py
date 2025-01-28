# -*- coding: utf-8 -*-
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


    ### models trained on 1% no label permutation
    path = r"C:\Users\pante\Desktop\Seizure_datasets\seizyml_train\normalize_per_file\trained_models_var_size"
    data = pd.read_csv(os.path.join(path, 'test_scores.csv'))
    data['percent_dataset_size'] = data['dataset_fraction']*100
    models_df = data.query("feature_set == 'best_10'")
    models_df = models_df[['model_name', 'id', 'fold', 'feature_set', 'percent_dataset_size']]
    models_df = get_importances(models_df, path, conds=['model_name'])
    plot_data = models_df.groupby(['model_name', 'features', 'percent_dataset_size']).sum(numeric_only=True).reset_index()
    plot_data['norm_importance'] = plot_data['importance']/plot_data.groupby(['model_name'])['importance'].transform('sum')

    # Pivot data to have percent_random_labels as columns and features as rows, for each model
    heatmap_data = {}
    linemap_data = {}
    reference = data['percent_dataset_size'].min()
    for model in plot_data['model_name'].unique():
        model_data = plot_data[plot_data['model_name'] == model]
        pivot_data = model_data.pivot(index='features', columns='percent_dataset_size', values='norm_importance')
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
        plt.xlabel("Percent Training Size")
        plt.ylabel("Features")
        plt.tight_layout()
        
    # plot absolute change
    for model, df in linemap_data.items():
        plt.figure(figsize=(12, 1))
        max_val = int(np.max([np.abs(df.min())] +[np.abs(df.max())])*100)/100
        df = pd.DataFrame([df.values])
        sns.heatmap(df, cmap="coolwarm_r", annot=False, fmt=".2f", cbar_kws={'label': 'Importance Difference'},
                    vmin=-max_val, vmax=max_val, linewidths=.1, xticklabels=df.index, yticklabels=False)
        plt.xlabel("PercentTraining Size")
    
