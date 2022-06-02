# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 09:12:51 2022

@author: fabia
"""

import joblib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def get_data(full_data=True):
    if full_data:
        filename = r"C:\Users\Fabian\Documents\Master Thesis Git\correctlyshrunk_all.joblib"
    else:
        filename = r"C:\Users\Fabian\Documents\Master Thesis Git\correctlyshrunk_part.joblib"
    with open(filename, "rb") as file:
        CV_Results = joblib.load(file)  
    estimators = [x for x in CV_Results.overall_stats.index if x.endswith("Y")]
    df_y = pd.DataFrame(columns=estimators)
    for estimator in df_y:
        results = []
        for repetition in CV_Results.rep_stats:
            results.append(repetition.loc[estimator, "overall"])
        df_y[estimator] = results
    global MEAN
    MEAN = CV_Results.overall_stats.loc["AllY", "overall"]
    return df_y
       

def makeplot(data, ax=None):
    sns.set_theme()
    percent_mean, percent_t4 = get_percentages(data)
    if ax:
        plot = sns.histplot(data, kde=True, bins=10, ax=ax)
        ax.axvline(MEAN, 0,0.95, color="red")
        ax.axvline(1970, 0,0.95, color="green")
        ax.set_title(f"{percent_mean} over mean, {percent_t4} over treatment 4")
    else:
        plot = sns.histplot(data, bins=10, kde=True)
        plt.axvline(MEAN, 0,0.95, color="red")
        plt.axvline(1970, 0,0.95, color="green")
        plot.fig.suptitle(f"{percent_mean} over mean, {percent_t4} over treatment 4")
    
def get_percentages(data):
    n = len(data)
    percent_mean = str(sum(data>=MEAN)) + "/" + str(n)
    percent_t4 = str(sum(data>1969.846698)) + "/" + str(n)
    return (percent_mean, percent_t4)

if __name__=="__main__":
    df_y = get_data()
    for column in df_y:
        print(column)
        makeplot(df_y[column])
        get_percentages(df_y[column])


