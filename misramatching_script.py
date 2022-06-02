# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:04:09 2022

@author: wmd975
"""

import pandas as pd
from mlmethods import *

END_OF_X = 48
START_OF_X = 2
TREAT_NAME = "treat"
Y_NAME = "buttonpresses"
CONTROL_INDEX = 7
PATH = r"C:\Users\fabia\Documents\Masterarbeit\Masterarbeit\temp\csv\expdata.csv"

params_allmodels = {
        "VirtualTwinRF": {
            "treat": {
                "treat_1": {
                    "n_estimators": 1000,
                    "max_features": 0.6,
                    "max_samples": 0.5,
                    "min_samples_leaf": 5,
                },
                "treat_2": {
                    "n_estimators": 1000,
                    "max_features": 0.7,
                    "max_samples": 0.4,
                    "min_samples_leaf": 5,
                },
                "treat_3": {
                    "n_estimators": 1000,
                    "max_features": 0.8,
                    "max_samples": 0.2,
                    "min_samples_leaf": 5,
                },
                "treat_4": {
                    "n_estimators": 1000,
                    "max_features": 1.0,
                    "max_samples": 0.3,
                    "min_samples_leaf": 10,
                },
                "treat_5": {
                    "n_estimators": 1000,
                    "max_features": 1.0,
                    "max_samples": 0.2,
                    "min_samples_leaf": 5,
                },
                "treat_6": {
                    "n_estimators": 1000,
                    "max_features": 0.6,
                    "max_samples": 0.5,
                    "min_samples_leaf": 5,
                },
            },
            "control": {
                "treat_1": {
                    "n_estimators": 1000,
                    "max_features": 1.0,
                    "max_samples": 0.3,
                    "min_samples_leaf": 10,
                },
                "treat_2": {
                    "n_estimators": 1000,
                    "max_features": 1.0,
                    "max_samples": 0.5,
                    "min_samples_leaf": 10,
                },
                "treat_3": {
                    "n_estimators": 1000,
                    "max_features": 1.0,
                    "max_samples": 0.3,
                    "min_samples_leaf": 10,
                },
                "treat_4": {
                    "n_estimators": 1000,
                    "max_features": 1.0,
                    "max_samples": 0.4,
                    "min_samples_leaf": 10,
                },
                "treat_5": {
                    "n_estimators": 1000,
                    "max_features": 0.8,
                    "max_samples": 0.3,
                    "min_samples_leaf": 10,
                },
                "treat_6": {
                    "n_estimators": 1000,
                    "max_features": 0.9,
                    "max_samples": 0.3,
                    "min_samples_leaf": 10,
                },
            },
            "main": {
                "treat_1": {
                    "n_estimators": 1000,
                    "max_features": 1.0,
                    "max_samples": 0.5,
                    "min_samples_leaf": 50,
                },
                "treat_2": {
                    "n_estimators": 1000,
                    "max_features": 1.0,
                    "max_samples": 0.5,
                    "min_samples_leaf": 50,
                },
                "treat_3": {
                    "n_estimators": 1000,
                    "max_features": 1.0,
                    "max_samples": 0.5,
                    "min_samples_leaf": 50,
                },
                "treat_4": {
                    "n_estimators": 1000,
                    "max_features": 1.0,
                    "max_samples": 0.5,
                    "min_samples_leaf": 50,
                },
                "treat_5": {
                    "n_estimators": 1000,
                    "max_features": 1.0,
                    "max_samples": 0.5,
                    "min_samples_leaf": 50,
                },
                "treat_6": {
                    "n_estimators": 1000,
                    "max_features": 1.0,
                    "max_samples": 0.5,
                    "min_samples_leaf": 50,
                },
            },
        },
        "CausalForestDML": {
            "treat_1": {
                "max_features": 0.2,
                "max_samples": 0.5,
                "min_samples_leaf": 20,
            },
            "treat_2": {
                "max_features": 1.0,
                "max_samples": 0.5,
                "min_samples_leaf": 5,
            },
            "treat_3": {
                "max_features": 0.9,
                "max_samples": 0.5,
                "min_samples_leaf": 5,
            },
            "treat_4": {
                "max_features": 0.8,
                "max_samples": 0.5,
                "min_samples_leaf": 5,
            },
            "treat_5": {
                "max_features": 0.3,
                "max_samples": 0.4,
                "min_samples_leaf": 10,
            },
            "treat_6": {
                "max_features": 0.3,
                "max_samples": 0.5,
                "min_samples_leaf": 5,
            },
        },
        "CausalNet": {"treat_1":{"params":{"optimizer":"GradientDescent","learning_rate":0.01,"alpha":0.1,"r_par":0.6,"hidden_layer_sizes":[20],"dropout_rates":[0.5],"batch_size":None,"max_epochs_without_change":60,"max_nepochs":10000,"seed":42,"verbose":False},"score":-8597310661459970.0},"treat_2":{"params":{"optimizer":"GradientDescent","learning_rate":0.01,"alpha":0.01,"r_par":1.0,"hidden_layer_sizes":[20],"dropout_rates":[0.5],"batch_size":None,"max_epochs_without_change":60,"max_nepochs":10000,"seed":42,"verbose":False},"score":-13484863488.000002},"treat_3":{"params":{"optimizer":"GradientDescent","learning_rate":0.01,"alpha":0.1,"r_par":1.0,"hidden_layer_sizes":[20],"dropout_rates":[0.5],"batch_size":None,"max_epochs_without_change":60,"max_nepochs":10000,"seed":42,"verbose":False},"score":-94330435928063.95},"treat_4":{"params":{"optimizer":"GradientDescent","learning_rate":0.01,"alpha":0.01,"r_par":0.0,"hidden_layer_sizes":[34],"dropout_rates":[0.5],"batch_size":None,"max_epochs_without_change":60,"max_nepochs":10000,"seed":42,"verbose":False},"score":-583906340372479.6},"treat_5":{"params":{"optimizer":"GradientDescent","learning_rate":0.01,"alpha":0.1,"r_par":0.6,"hidden_layer_sizes":[100],"dropout_rates":[0.5],"batch_size":None,"max_epochs_without_change":60,"max_nepochs":10000,"seed":42,"verbose":False},"score":-565716675723265.4},"treat_6":{"params":{"optimizer":"GradientDescent","learning_rate":0.01,"alpha":0.01,"r_par":0.0,"hidden_layer_sizes":[60],"dropout_rates":[0.5],"batch_size":None,"max_epochs_without_change":60,"max_nepochs":10000,"seed":42,"verbose":False},"score":-444754936463360.1}},
        "DoubleRobust": {
            "treat_1": {
                "max_features": 0.3,
                "max_samples": 0.5,
                "min_samples_leaf": 20,
                "min_weight_fraction_leaf": 0.0,
                "mc_iters": 5,
            },
            "treat_2": {
                "max_features": 0.4,
                "max_samples": 0.5,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0.0,
                "mc_iters": 3,
            },
            "treat_3": {
                "max_features": 0.6,
                "max_samples": 0.5,
                "min_samples_leaf": 10,
                "min_weight_fraction_leaf": 0.0,
                "mc_iters": 3,
            },
            "treat_4": {
                "max_features": 0.1,
                "max_samples": 0.5,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0.0,
                "mc_iters": 3,
            },
            "treat_5": {
                "max_features": 0.3,
                "max_samples": 0.5,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0.0,
                "mc_iters": 3,
            },
            "treat_6": {
                "max_features": 0.7,
                "max_samples": 0.4,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0.0,
                "mc_iters": 3,
            },
        },
    }

def getMinMax(df_list):
    min_df = df_list[0].copy()
    max_df = df_list[0].copy()
    for rowIndex, row in min_df.iterrows():
        for columnIndex, vlaue in row.items():
            min_df.loc[rowIndex,columnIndex] = min(
                [df.loc[rowIndex,columnIndex] for df in df_list])
            max_df.loc[rowIndex,columnIndex] = max(
                [df.loc[rowIndex,columnIndex] for df in df_list])
    return (min_df, max_df)

def getTreatDiff(stats):
    return round(stats.iloc[:-3,-1][::2].max() - stats.iloc[-2].max())

df = pd.read_csv(PATH)
used_treatments = ['treat_1', 'treat_2', 'treat_3', 'treat_4', 'treat_5', 'treat_6'] # 
Matcher = MisraMatching(params_allmodels)
CV_Results = Matcher.cross_validate(
    data=df,
    repetitions=3,
    folds=3,
    treatment_names=used_treatments,
    used_estimators=[
        DoubleRobustHTE,
        VirtualTwinRF,
        CausalNets,
        CausalForestHTE,
    ],
)


min_df, max_df = getMinMax(CV_Results.rep_stats)
    
writer = pd.ExcelWriter(r"V:\MTurk\MisraMatching\Overview_files\T4-6_res.xlsx", engine='xlsxwriter')
CV_Results.overall_stats.to_excel(writer, sheet_name='Overall')
min_df.to_excel(writer, sheet_name='Minimum')               
max_df.to_excel(writer, sheet_name='Maximum')

writer.save()