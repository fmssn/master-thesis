# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:00:39 2022

@author: fabia
"""
import pandas as pd 
import json
from mlmethods import *

possible_treatments = ["treat_" + str(i) for i in range(1, 7)]
df = pd.read_csv(
    r"C:\Users\fabia\Documents\Masterarbeit\Masterarbeit\temp\csv\expdata.csv")


def save_params(name:str, params:dict):
    filename =\
        'C:/Users/fabia/Documents/Masterarbeit/Masterarbeit/CV/'+ name + '.txt'
    with open(filename, 'w') as file:
        file.write(json.dumps(params))


CausalN = CausalNets(possible_treatments)
cn_grid = [
    {'optimizer': optimizer, 'learning_rate': learning_rate,
        'alpha': alpha, 'r_par': r_par,
        'hidden_layer_sizes':hidden_layer_size,
        'dropout_rates':dropout_rate, 
        'batch_size':None,
        'max_epochs_without_change':60, 
        'max_nepochs':10000, 
        'seed':RS, 
        'verbose':False         
        }
    for optimizer in ['Adam', 'GradientDescent', 'RMSprop']
    for hidden_layer_size, dropout_rate in (
        [[60],[0.5]],
        [[100],[0.5]],
        [[30, 20],[0.5, 0]],
        [[30, 10],[0.3, 0.1]],
        [[30, 30],[0, 0]],
        [[30, 30],[0.5, 0]],
        [[100, 30, 20],[0.5, 0.5, 0]],
        [[80, 30, 20],[0.5, 0.5, 0]],
        [[20],[0.5]],
        [[34],[0.5]],
        [[10, 7],[0.5, 0]],
        [[10, 4],[0.3, 0.1]],
        [[10, 10],[0, 0]],
        [[10, 10],[0.5, 0]],
        [[34, 10, 7],[0.5, 0.5, 0]],
        [[27, 10, 7],[0.5, 0.5, 0]]
        )
    for learning_rate in [0.1, 0.05, 0.01]
    for alpha in [0.01, 0.1, 1.]
    for r_par in [0., 0.3, 0.6, 1.]
]
CausalN.cross_validate(data=df, gridsearch_params=cn_grid)
save_params('CausalNet', CausalN.optimal_params)




CausalF = CausalForestHTE(possible_treatments)
cf_grid = [
    {'max_features': max_features,
     'max_samples': max_samples,
     'min_samples_leaf': min_samples_leaf,
     'min_var_fraction_leaf': min_var_fraction_leaf,
     'max_depth': max_depth}
    for max_features in [i/10. for i in range(2, 11)]
    for max_samples in [i/10. for i in range(1, 6)]
    for min_samples_leaf in [5, 10, 20, 50]
    for min_var_fraction_leaf in [None, .1, .2, .3, .4]
    for max_depth in [5, 10, 25, 50, 75, 100, None]
]
CausalF.cross_validate(df, cf_grid)
save_params('CausalForest', CausalF.optimal_params)



RandomF = VirtualTwinRF(possible_treatments)
rf_grid = [
    {'max_features': max_features, 'max_samples': max_samples,
        'min_samples_leaf': min_samples_leaf, 'max_depth': max_depth}
    for max_features in [i/10. for i in range(2, 11)]
    for max_samples in [i/10. for i in range(1, 6)]
    for min_samples_leaf in [2, 5, 10, 20, 50]
    for max_depth in [5, 10, 25, 50, 75, 100, None]
]
RandomF.cross_validate_Intermediate(df, rf_grid)
RandomF.cross_validate_Main(df, rf_grid)
save_params("VirtualTwin", RandomF.optimal_params)


