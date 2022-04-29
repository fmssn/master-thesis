# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:54:08 2022

@author: fabia
"""

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import warnings
import copy
import random

from sklearn.model_selection import train_test_split, RepeatedKFold, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from econml.grf import CausalForest
from econml.dml import CausalForestDML
from econml.dr import ForestDRLearner
from causal_nets.utils import CoeffNet
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler

import time

END_OF_X = 48
START_OF_X = 2
SEED = 42
TREAT_NAME = "treat"
Y_NAME = "buttonpresses"
CONTROL_INDEX = 7
RS = 42

        
def printfull(df):
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)
        
def standardise(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

def getTreatmentSnippet(data, treatment_name):
    '''
    Get dataframe snippet with only control and respective treatment

    Parameters
    ----------
    data : pd.DataFrame
        Contains the data.
    treatment_name : str
        Name of the treatment, e.g. 'treat_2'.

    Returns
    -------
    dataT : pd.DataFrame
        Dataframe only containing observations with the the respective
        treatment and the control variable, not with other treatments.
    '''
    dataT = data.loc[
        (
            (data[TREAT_NAME] == CONTROL_INDEX)
            | (data[TREAT_NAME] == int(treatment_name[-1]))
        )
    ]
    return dataT


def t_risk(data_train, data_test, treatment, tau_pred):
    '''
    [(Y - M(X) ) - (W - P(X) )*tau_pred ]^2
     
    Y: real values
    M(X): E[Y|X] e.g. from regression or lasso
         
    W: Treated oder Control (T)
    P(X): Propensity Score
    '''
    X_test, Y_test, T_test = getXY(data_test, treatment)
    X_train, Y_train, T_train = getXY(data_train, treatment)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model_m = LassoCV(max_iter=1000)
    model_m.fit(X=X_train, y=Y_train)
    
    model_p = LogisticRegressionCV(penalty='l1', max_iter=1000, solver='liblinear')
    model_p.fit(X=X_train, y=T_train)
    
    m = model_m.predict(X_test)
    p = model_p.predict(X_test)
    
    score = sum ( np.square(( Y_test - m ) - ( T_test - p ) * tau_pred ))
    return score

class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def track(self, text=None):
        now = time.perf_counter()
        if text:
            print(text)
        time_passed = now - self.start
        minutes = int(time_passed / 60)
        seconds = int(round(time_passed % 60))
        print(f"It took {minutes} minutes and {seconds} seconds.")
        self.start = now
        
class ShrinkageEstimators:
    def __init__(self, data):
        self.treatments = [1,2,3,4,5,6] 
        self.data = data
        self.getATEvar()
        self.getTauVars()
        self.ate = data[data["treat"].isin(self.treatments)].buttonpresses.sum()

    def shrinkJamesStein(self, preds, towards_overall_mean=False):
        """Shrink predictions in line with James Stein estimator.

        Parameters
        ----------
        preds : pd.Series
            The predictions for the individual treatment effects

        towards_overall_mean : bool, optional
            Whether to shrink towards the overall average treatment effect of
            all treatments (or towards the mean of the respective treatment if
            False). The default is False.

        Returns
        -------
        preds_altered : pd.Series
            Shrunk predictions for individual treatment effects.
        c : float
            Shrinking factor.
        """
        n = len(preds)
        c = 1 - (n - 3) * self.var_ate / \
            (np.square(preds - preds.mean()).sum())
        if towards_overall_mean:
            preds_altered = (preds - self.ate) * c + self.ate
        else:
            preds_altered = (preds - preds.mean()) * c + preds.mean()
        return preds_altered, c

    def shrinkBiasAdjustment(self, preds, towards_overall_mean=False):
        """Shrink predictions according the Chen/Zimmermann shrinkage estimator.

        Parameters
        ----------
        preds : pd.Series
            The predictions for the individual treatment effects

        Returns
        -------
        preds_alt: pd.Series
            Shrunk predictions for individual treatment effects.
        c : float
            Shrinking factor
        """
        c = self.var_ate / (self.var_ate + self.tau_vars)
        if towards_overall_mean:
            preds_alt = c * preds + (1 - c) * self.ate
        else:
            preds_alt = c * preds + (1 - c) * preds.mean()
        return preds_alt, c

    def getTauVars(self):
        """Get the variance of all 6 treatments.

        Measured by the respective variance of the OLS estimator in a regression
        Y ~ beta_0 + beta_1 * Treated(J)_i, where Treated(J)_i is 1 if the
        individual is treated with treatment J and 0 if not (control). The
        regression is performed on a cutout dataset only including control and
        observations with treatment J, respectively.
        """
        data = self.data
        tau_vars = []
        for t in self.treatments:
            treatment = "treat_" + str(t)
            data_snippet = data[(data["treat"] == t) | (data["treat"] == 7)]
            model = smf.ols(
                "buttonpresses ~ {}".format(treatment), data=data_snippet
            ).fit()
            tau_vars.append(model.bse[1] ** 2)
        self.tau_vars = tau_vars
        return self

    def getATEvar(self):
        """Performs a OLS regression to retrieve variance of the ATE.

        The regression is Y ~ beta_0 + beta_1 * Treated_i + e, where Treated_i
        is 1 if the individual is treated at all by any treatment and 0 if not.
        """
        T = self.data["treat"] != CONTROL_INDEX
        T = T.astype("int")
        data = pd.DataFrame({"Y": self.data["buttonpresses"], "treat": T})
        data.Y = pd.to_numeric(data.Y, errors="coerce")
        data.treat = pd.to_numeric(data.treat, errors="coerce")
        model = smf.ols("Y ~ T", data=data).fit()
        self.var_ate = model.bse["T"] ** 2
        return self

    def shrink(self, tau_preds, prefix="shrunk"):
        """Shrink predictions with all four shrinkage methods.

        Parameters
        ----------
        tau_preds : pd.DataFrame
            containts
        prefix : TYPE, optional
            DESCRIPTION. The default is 'shrunk'.

        Returns
        -------
        df_summary : pd.DataFrame
            Contains Misra-Matched treatment assignments. Columns are treatment
            + overall, rows are Y values and number of matched Ys.
        """

        tau_alt, __ = self.shrinkJamesStein(tau_preds)
        df_matched_alt = MisraMatching.getMatchingPredictions(
            self.data, tau_alt, prefix + "_JS_individ"
        )

        tau_alt2, __ = self.shrinkJamesStein(
            tau_preds, towards_overall_mean=True)
        df_matched_alt2 = MisraMatching.getMatchingPredictions(
            self.data, tau_alt2, prefix + "_JS_pool"
        )

        predsBAdj, __ = self.shrinkBiasAdjustment(tau_preds)
        df_matched_BAdj = MisraMatching.getMatchingPredictions(
            self.data, predsBAdj, prefix + "_BAdj_individ"
        )

        predsBAdj2, __ = self.shrinkBiasAdjustment(
            tau_preds, towards_overall_mean=True)
        df_matched_BAdj2 = MisraMatching.getMatchingPredictions(
            self.data, predsBAdj2, prefix + "_BAdj_pool"
        )

        df_summary = pd.concat(
            [df_matched_alt, df_matched_alt2, df_matched_BAdj, df_matched_BAdj2]
        )

        return df_summary


def getXY(data, treatment=None):
    """
    Split X, Y (and T) values from the main imported data.

    Parameters
    ----------
    data : pd.DataFrame
        Contains the formated data from round 1 of the experiment.
    index_t : int, optional
        Whether and what treatment dummy vector to additionally return.
        The default is None.

    Returns
    -------
    X, Y, (T): pd.Dataframe, pd.Series, pd Series
        Respective X, Y (and T) dataframes / vectors from inputed dataframe.
    """
    X = data.iloc[:, START_OF_X:END_OF_X]
    Y = data[Y_NAME]
    if treatment:
        index_t = data.columns.get_loc(treatment)
        T = data.iloc[:, index_t]
        return X, Y, T
    else:
        return X, Y


class HTEestimator(ABC):
    
    def __init__(self, treatment_names):
        self.models = dict(zip(treatment_names, [None] * len(treatment_names)))

    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def predict():
        pass

    def cross_validate(self, data, gridsearch_params, folds=3, random_search=None):
        self.optimal_params = {
            treatment: dict(
                params=None, score=10**10) for treatment in self.models.keys()}
        fold_iterator = KFold(folds)
        i = 0
        if random_search:
            if 0 < random_search < 1:
                k = int(len(gridsearch_params) * random_search)
            elif random_search >=1:
                k = random_search
            else:
                raise ValueError("random_search has to be >0 but finite.")
            gridsearch_params = random.sample(
                    gridsearch_params, k)
        n = len(gridsearch_params)
        for treatment, __ in self.models.items():
            for params in gridsearch_params:
                i += 1
                for train, test in fold_iterator.split(data):
                    data_train = getTreatmentSnippet(
                        data.iloc[train], treatment)
                    data_test = getTreatmentSnippet(
                        data.iloc[test], treatment)

                    tau_pred = self.trainPredictSingleTreatment(
                        params, data_train, data_test, treatment)

                    score = t_risk(data_train, data_test, treatment, tau_pred)
                    if score < self.optimal_params[treatment]['score']:
                        self.optimal_params[treatment]['params'] = params
                        self.optimal_params[treatment]['score'] = score
                print(f'Finished {i}/{n}')
            print(f'Finished training treatment {treatment}.')
            print('Best params were:')
            print(self.optimal_params[treatment]['params'])     


class VirtualTwinRF(HTEestimator):
    __name__ = "VirtualTwinRF"

    @staticmethod
    def getMSE(random_forest, X, Y):
        y_pred = random_forest.predict(X)
        mse = np.mean( np.square(y_pred - Y))
        return mse
    
    @staticmethod
    def splitTreatControl(data, treatment):
        data_untreated = data.loc[(data[TREAT_NAME]==CONTROL_INDEX)]
        data_treated = data.loc[(data[treatment]==1)]
        X_treated, Y_treated = getXY(data_treated)
        X_untreated, Y_untreated = getXY(data_untreated)
        return X_treated, Y_treated, X_untreated, Y_untreated
        
    def cross_validate_Intermediate(self, data, gridsearch_params, folds=3):
        self.optimal_params = {
            treatment:{
                model: dict(params=None, score=10**10) for model in ['treat', 'control', 'main']}
            for treatment in self.models.keys()}
        fold_iterator = KFold(folds)
        n = len(gridsearch_params)
        for treatment, __ in self.models.items():
            i = 0
            for params in gridsearch_params:
                i += 1
                mse_treat_list = []
                mse_untreated_list = []
                for train, test in fold_iterator.split(data):
                    data_train = getTreatmentSnippet(
                        data.iloc[train], treatment)
                    data_test = getTreatmentSnippet(
                        data.iloc[train], treatment)

                    X_train_treated, Y_train_treated,\
                        X_train_untreated, Y_train_untreated =\
                            self.splitTreatControl(data_train, treatment)
                    
                    X_test_treated, Y_test_treated,\
                        X_test_untreated, Y_test_untreated =\
                            self.splitTreatControl(data_test, treatment)
                    
                    model_treat = RandomForestRegressor(
                        n_estimators=1000, random_state=RS, **params)
                    model_treat.fit(X=X_train_treated, y=Y_train_treated)
                    mse_treat = self.getMSE(model_treat, X_test_treated, Y_test_treated)
                    mse_treat_list.append(mse_treat)
                    
                    model_untreated = RandomForestRegressor(
                        n_estimators=1000, random_state=RS, **params)
                    model_untreated.fit(X=X_train_untreated, y=Y_train_untreated)
                    mse_untreated = self.getMSE(model_untreated, X_test_untreated, Y_test_untreated)
                    mse_untreated_list.append(mse_untreated)
                
                    
                overall_mse_treat = np.mean(mse_treat_list)
                overall_mse_untreated = np.mean(mse_untreated_list)
                
                if overall_mse_treat < self.optimal_params[treatment]['treat']['score']:
                    self.optimal_params[treatment]['treat']['params'] = params
                    self.optimal_params[treatment]['treat']['score'] = overall_mse_treat
                    
                if overall_mse_untreated < self.optimal_params[treatment]['control']['score']:
                    self.optimal_params[treatment]['control']['params'] = params
                    self.optimal_params[treatment]['control']['score'] = overall_mse_untreated
                        
                print(f'Finished {i}/{n}')
            print(f'Finished training treatment {treatment}.')
            print('Best params were:')
            print(self.optimal_params[treatment]['treat'])
            print(self.optimal_params[treatment]['control'])
            
    def cross_validate_Main(self, data, gridsearch_params, folds = 3): 
        fold_iterator = KFold(folds)
        n = len(gridsearch_params)
        for treatment, __ in self.models.items():
            i = 0
            for params in gridsearch_params:
                i += 1
                mse_list = []
                for train, test in fold_iterator.split(data):
                    data_train = getTreatmentSnippet(
                        data.iloc[train], treatment)
                    data_test = getTreatmentSnippet(
                        data.iloc[train], treatment)

                    X_train_treated, Y_train_treated,\
                        X_train_untreated, Y_train_untreated =\
                            self.splitTreatControl(data_train, treatment)
                    
                    X_test_treated, Y_test_treated,\
                        X_test_untreated, Y_test_untreated =\
                            self.splitTreatControl(data_test, treatment)
                    
                    model_treat = RandomForestRegressor(
                        n_estimators=1000, random_state=RS,
                        **self.optimal_params[treatment]['treat']['params'])
                    model_treat.fit(X=X_train_treated, y=Y_train_treated)
                    
                    model_untreated = RandomForestRegressor(
                        n_estimators=1000, random_state=RS,
                        **self.optimal_params[treatment]['control']['params'])
                    model_untreated.fit(X=X_train_untreated, y=Y_train_untreated)
                    
                    # Training the model for predicting the y1 for the untreated
                    model_treat.fit(X=X_train_treated, y=Y_train_treated)
                    Y1_pred_train_untreated = model_treat.predict(X_train_untreated)
                    Y1_pred_test_untreated = model_treat.predict(X_test_untreated)

                    # Training the model for predicting the y0 for the treated
                    model_untreated.fit(X=X_train_untreated, y=Y_train_untreated)
                    Y0_pred_train_treated = model_untreated.predict(X_train_treated)
                    Y0_pred_test_treated = model_untreated.predict(X_test_treated)

                    # Constructing the treatment effect
                    tau_pred_train=pd.concat(
                        [Y_train_treated-Y0_pred_train_treated,
                         Y1_pred_train_untreated-Y_train_untreated])

                    # Training the model for predicting the treatment effect
                    X_train = pd.concat([X_train_treated, X_train_untreated])
                    model_main=RandomForestRegressor(
                        n_estimators=1000, random_state=RS, **params)
                    model_main.fit(X=X_train, y=tau_pred_train)
                    
                    # Creating the MSE score
                    tau_pred_test = pd.concat(
                        [Y_test_treated-Y0_pred_test_treated,
                         Y1_pred_test_untreated-Y_test_untreated])
                    
                    X_test = pd.concat([X_test_treated, X_test_untreated])
                    
                    mse = self.getMSE(model_main, X_test, tau_pred_test)
                    mse_list.append(mse)
                    
                overall_mse = np.mean(mse_list)
                
                if overall_mse < self.optimal_params[treatment]['main']['score']:
                    self.optimal_params[treatment]['main']['params'] = params
                    self.optimal_params[treatment]['main']['score'] = overall_mse

                        
                print(f'Finished {i}/{n}')
            print(f'Finished training treatment {treatment}.')
            print('Best params were:')
            print(self.optimal_params[treatment]['main'])



    def fit(self, data, params=dict()):
        """
        Trains VTRF models on all six treatments for given data.

        Parameters
        ----------
        data : pd.dataframe
            Dataframe containing X, T and Y variables. Need to be of specific form
            to work.
        params : dict of dicts, optional
            Contains the (optimal) parameters for each the three random forest
            models needed for the approach and for each of the different treatments,
            respectively. Needed format is:
                dict(treat={dict of arguments},
                     control={dict of arguments},
                     main={dict of arguments})

        """
        for key, __ in self.models.items():

            # Formatting data from the training split
            data_treat = data.loc[(data[TREAT_NAME] == int(key[-1]))]
            data_control = data.loc[(data[TREAT_NAME] == CONTROL_INDEX)]
            X_treat, Y_treat = getXY(data_treat)
            X_control, Y_control = getXY(data_control)

            # Init the the models
            model_treat, model_control, model_main = self.init_models(
                params, key)

            # Training the model for predicting the y1 for the untreated
            model_treat.fit(X=X_treat, y=Y_treat)
            Y1_pred_control = model_treat.predict(X_control)

            # Training the model for predicting the y0 for the treated
            model_control.fit(X=X_control, y=Y_control)
            Y0_pred_treat = model_control.predict(X_treat)

            # Constructing the treatment effect
            tau_pred_train = Y_treat - Y0_pred_treat
            tau_pred_train = pd.concat(
                [tau_pred_train, Y1_pred_control - Y_control])

            # Training the model for predicting the treatment effect
            X_train = pd.concat([X_treat, X_control])
            model_main.fit(X=X_train, y=tau_pred_train)
            self.models[key] = model_main
        return self

    def init_models(self, params, key):
        """
        Initialize all three models for the Virtual Twin RF method.

        Parameters
        ----------
        params : dict of dicts
            Contains the´
        key : TYPE
            DESCRIPTION.

        Returns
        -------
        model_treat : RandomForestRegressor
            Will be trained on the treated for predicting the Y1 outcome for
            the untreated.
        model_control : RandomForestRegressor
            Will be trained on the untreated for predicting the Y0 outcome for
            the treated.
        model_main : RandomForestRegressor
            For predicting the

        """
        if params:
            model_treat = RandomForestRegressor(
                random_state=RS, **params["treat"][key])
            model_control = RandomForestRegressor(
                random_state=RS, **params["control"][key])
            model_main = RandomForestRegressor(
                random_state=RS, **params["main"][key])
        else:
            model_treat = RandomForestRegressor(random_state=RS)
            model_control = RandomForestRegressor(random_state=RS)
            model_main = RandomForestRegressor(random_state=RS)
        return model_treat, model_control, model_main

    def predict(self, X):
        """
        Predict HTE for all six treatments given X.


        Parameters
        ----------
        X : pd.DataFrame
            Contains the covariates in the same format as used for the training.

        Returns
        -------
        pred_df : pd.DataFrame
            Contains the predicted individual treatment effect for all treatments
            (columns) for all observations of the inputed X dataframe(rows).

        """
        pred_df = pd.DataFrame()
        for treatment, model in self.models.items():
            tau_pred = model.predict(X)
            pred_df[treatment] = tau_pred
        return pred_df

    def cross_validate(self):
        return self





class CausalForestHTE(HTEestimator):
    __name__ = "CausalForest"

    def trainPredictSingleTreatment(self, params, data_train, data_test, treatment):
        X_train, Y_train, T_train = getXY(
            data_train, treatment)
        X_test, Y_test, T_test = getXY(
            data_test, treatment)

        model = CausalForest(
            n_estimators=1000, criterion='mse', random_state=RS, **params)
        model.fit(X=X_train, T=T_train, y=Y_train)

        tau_pred = np.concatenate(model.predict(X_test))
        return tau_pred

    def fit(self, data, params=dict()):
        """
        Trains the Causal Forests model(s) on the given data with the given
        parameters.

        Parameters
        ----------
        data : pd.dataframe
            Dataframe containing X, T and Y variables. Need to be of specific form
            to work.
        params_dict : dict of dicts
            Contains the (optimal) parameters for each of the treatments.

        Returns
        -------
        model_dict : dict of models
            Contains the trained models for each treatment.

        """
        for key, __ in self.models.items():
            data1 = data.loc[
                (
                    (data[TREAT_NAME] == CONTROL_INDEX)
                    | (data[TREAT_NAME] == int(key[-1]))
                )
            ]
            X, Y, T = getXY(data1, key)

            model = CausalForest(n_estimators=1000, criterion="mse")
            with warnings.catch_warnings():  # For suppressing convergence fails
                warnings.simplefilter("ignore")
                model.fit(X=X, T=T, y=Y)

            self.models[key] = model
        return self
    

    def predict(self, X):
        """Predict HTE for all six treatments given X.


        Parameters
        ----------
        X : pd.DataFrame
            Contains the covariates in the same format as used for the training.

        Returns
        -------
        pred_df : pd.DataFrame
            Contains the predicted individual treatment effect for all treatments
            (columns) for all observations of the inputed X dataframe(rows).

        """
        pred_df = pd.DataFrame()
        for treatment, model in self.models.items():
            tau_pred = model.predict(X)
            pred_df[treatment] = np.concatenate(tau_pred)
        return pred_df



class CausalForestDML6(HTEestimator):
    __name__ = "CausalForestDML"

    def fit(self, data, params=dict()):
        """
        Trains the Causal Forests DML model(s) on the given data with the given
        parameters.

        Parameters
        ----------
        data : pd.dataframe
            Dataframe containing X, T and Y variables. Need to be of specific form
            to work.
        params_dict : dict of dicts
            Contains the (optimal) parameters for each of the treatments.

        Returns
        -------
        model_dict : dict of models
            Contains the trained models for each treatment.

        """
        for key, __ in self.models.items():
            data1 = data.loc[
                (
                    (data[TREAT_NAME] == CONTROL_INDEX)
                    | (data[TREAT_NAME] == int(key[-1]))
                )
            ]
            X, Y, T = getXY(data1, key)

            scaler = StandardScaler()            

            model = CausalForestDML(
                n_estimators=1000, random_state=RS, 
                featurizer = scaler, **params[key])
            with warnings.catch_warnings():  # For suppressing convergence fails
                warnings.simplefilter("ignore")
                model.fit(X=X, T=T, Y=Y)

            self.models[key] = model

        return self

    def predict(self, X):
        """Predict HTE for all six treatments given X.


        Parameters
        ----------
        X : pd.DataFrame
            Contains the covariates in the same format as used for the training.

        Returns
        -------
        pred_df : pd.DataFrame
            Contains the predicted individual treatment effect for all treatments
            (columns) for all observations of the inputed X dataframe(rows).

        """
        pred_df = pd.DataFrame()
        for treatment, model in self.models.items():
            tau_pred = model.effect(X)
            pred_df[treatment] = tau_pred.reshape(-1)
        return pred_df

    def cross_validate(self):
        return self


class DoubleRobustHTE(CausalForestDML6):
    __name__ = "DoubleRobust"

    def fit(self, data, params=dict()):
        """
        Trains the Double Robust DML model(s) on the given data with the given
        parameters.

        Parameters
        ----------
        data : pd.dataframe
            Dataframe containing X, T and Y variables. Need to be of specific form
            to work.
        params_dict : dict of dicts
            Contains the (optimal) parameters for each of the treatments.

        Returns
        -------
        model_dict : dict of models
            Contains the trained models for each treatment.

        """
        for key, __ in self.models.items():
            data1 = data.loc[
                (
                    (data[TREAT_NAME] == CONTROL_INDEX)
                    | (data[TREAT_NAME] == int(key[-1]))
                )
            ]
            X, Y, T = getXY(data1, key)
            X = standardise(X)

            model = ForestDRLearner(n_estimators=1000, random_state=RS, **params[key])
            with warnings.catch_warnings():  # For suppressing convergence fails
                warnings.simplefilter("ignore")
            model.fit(X=X, T=T, Y=Y)

            self.models[key] = model
        return self
    
    def predict(self, X):
        """Predict HTE for all six treatments given X.


        Parameters
        ----------
        X : pd.DataFrame
            Contains the covariates in the same format as used for the training.

        Returns
        -------
        pred_df : pd.DataFrame
            Contains the predicted individual treatment effect for all treatments
            (columns) for all observations of the inputed X dataframe(rows).

        """
        pred_df = pd.DataFrame()
        X = standardise(X)
        for treatment, model in self.models.items():
            tau_pred = model.effect(X)
            pred_df[treatment] = tau_pred.reshape(-1)
        return pred_df


class CausalNets(HTEestimator):
    __name__ = "CausalNet"

    def trainPredictSingleTreatment(
            self, params, data_train, data_test,  treatment, return_score=True):
        data_train, data_valid = train_test_split(
            data_train, test_size=0.3, random_state=RS)

        X_train, Y_train, T_train = getXY(data_train, treatment)

        X_valid, Y_valid, T_valid = getXY(data_valid, treatment)

        X_test, Y_test, T_test = getXY(data_test, treatment)

        model = CoeffNet(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        betas_model, history = model.training_NN(
            training_data=[X_train, T_train, Y_train],
            validation_data=[X_valid, T_valid, Y_valid],
        )
        print(min(history["val_loss"]))
        print("stop pls")

        for treatment, net in self.models.items():
            tau_pred, mu0pred = model.retrieve_coeffs(
                betas_model=betas_model, input_value=X_test
            )
            tau_pred = np.concatenate(tau_pred)
        if return_score:
            return tau_pred, history
        return tau_pred

    def fit(self, data, params={}):
        """
        Trains the Causal Net model(s) on the given data with the given
        parameters.

        Parameters
        ----------
        data : pd.dataframe
            Dataframe containing X, T and Y variables. Need to be of specific form
            to work.
        params_dict : dict of dicts
            Contains the (optimal) parameters for each of the treatments.

        """
        if not params:
            params = {
                "hidden_layer_sizes": [60, 30],
                "dropout_rates": [0, 0],
                "batch_size": None,
                "alpha": 0.0,
                "r_par": 0.0,
                "optimizer": "Adam",
                "learning_rate": 0.0009,
                "max_epochs_without_change": 30,
                "max_nepochs": 10000,
                "seed": RS,
                "verbose": False,
            }

        data_train, data_test = train_test_split(
            data, test_size=0.3, random_state=42)

        for treatment, __ in self.models.items():
            data1_train = getTreatmentSnippet(data_train, treatment)
            X_train, Y_train, T_train = getXY(data1_train, treatment)

            data1_test = getTreatmentSnippet(data_test, treatment)
            X_valid, Y_valid, T_valid = getXY(data1_test, treatment)
            
            model = CoeffNet(**params[treatment])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                betas_model, history = model.training_NN(
                    training_data=[X_train, T_train, Y_train],
                    validation_data=[X_valid, T_valid, Y_valid],
                )
                
            self.models[treatment] = {"betas_model": betas_model, "model": model}
        return self

    def predict(self, X):
        """Predict HTE for all six treatments given X.


        Parameters
        ----------
        X : pd.DataFrame
            Contains the covariates in the same format as used for the training.

        Returns
        -------
        pred_df : pd.DataFrame
            Contains the predicted individual treatment effect for all treatments
            (columns) for all observations of the inputed X dataframe(rows).

        """
        pred_df = pd.DataFrame()
        for treatment, net in self.models.items():
            tau_pred, mu0pred = net["model"].retrieve_coeffs(
                betas_model=net["betas_model"], input_value=X
            )
            pred_df[treatment] = np.concatenate(tau_pred)
        return pred_df
    
    def cross_validate(self, data, gridsearch_params, folds=3, random_search=None):
        self.optimal_params = {
            treatment: dict(
                params=None, score=10**10) for treatment in self.models.keys()}
        fold_iterator = KFold(folds)
        i = 0
        if random_search:
            if 0 < random_search < 1:
                k = int(len(gridsearch_params) * random_search)
            elif random_search >=1:
                k = random_search
            else:
                raise ValueError("random_search has to be >0 but finite.")
            gridsearch_params = random.sample(
                    gridsearch_params, k)
        n = len(gridsearch_params)
        for treatment, __ in self.models.items():
            for params in gridsearch_params:
                i += 1
                for train, test in fold_iterator.split(data):                   
                    data_train = getTreatmentSnippet(data.iloc[train], treatment)
                    X_train, Y_train, T_train = getXY(data_train, treatment)

                    data_valid, data_test = train_test_split(
                        data.iloc[test], test_size=0.5)
                
                    X_valid, Y_valid, T_valid = getXY(data_valid, treatment)
                    X_test, Y_test, T_test = getXY(data_test, treatment)
                    
                    model = CoeffNet(**params)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        betas_model, history = model.training_NN(
                            training_data=[X_train, T_train, Y_train],
                            validation_data=[X_valid, T_valid, Y_valid],
                        )
                    tau_pred, mu0pred = model.retrieve_coeffs(
                        betas_model=betas_model, input_value=X_test
                        )
                    predictions = np.concatenate(mu0pred) + np.concatenate(tau_pred) * T_test
                    score = sum(np.square(predictions - T_test))
                    if score < self.optimal_params[treatment]['score']:
                        self.optimal_params[treatment]['params'] = params
                        self.optimal_params[treatment]['score'] = score
                print(f'Finished {i}/{n*6}')
            print(f'Finished training treatment {treatment}.')
            print('Best params were:')
            print(self.optimal_params[treatment]['params'])  


class MisraMatching:
    def __init__(self, params_allmodels):
        self.params_allmodels = params_allmodels

    def getStatsForModel(
        self,
        ModelClass,
        data_train,
        data_test,
        treatment_names,
        Shrinker=None,
        return_assignments=False,
    ):
        """
        Generates a summary dataframe for Misra-Matched outcomes and respective
        counts.

        Parameters
        ----------
        ModelClass : HTEestimator
            Estimator for predicting the (six) heterogenous treatment effects.
            Must contain the methods fit() and predict().
        data_train : pd.DataFrame
            Contains the data to train on.
        data_test : pd.DataFrame
            Contains the data to test on.
        Shrinker : ShinkageClass, optional
            Shrinkage class which contains the shrink() method, returning the
            sum-summaries in line with the getMatchingPredictions Method.
            The default is None.
        return_assignments : Boolean, optional
            Whether to also return the prediction of the optimal treatment
            assignment for the model. The default is False.

        Returns
        -------
        summary_stats_df: pd.DataFrame
            Dataframe containing the sums of matched Y outcomes and the
            counts for each respective treatment as well as over all treatments.

        (Optional:)
        assignments: pd.Series
            Optimal treatment assignment for the respective test set.
        """
        model = ModelClass(treatment_names)
        if self.params_allmodels:
            model.fit(data_train, params=self.params_allmodels[model.__name__])
        else:
            model = model.fit(data_train)
        tau_pred = model.predict(data_test.iloc[:, START_OF_X:END_OF_X])
        summary_stats_df = MisraMatching.getMatchingPredictions(
            data_test, tau_pred, model.__name__
        )
        if Shrinker:
            shrunk_summaries = Shrinker.shrink(
                tau_pred, prefix=f"{model.__name__}_shrunk_"
            )
            summary_stats_df = pd.concat([summary_stats_df, shrunk_summaries])
        if return_assignments:
            assignments = tau_pred.idxmax(axis=1)
            assignments.rename(model.__name__, inplace=True)
            return summary_stats_df, assignments
        else:
            return summary_stats_df

    def cross_validate(
        self,
        data,
        treatment_names,
        repetitions=1,
        folds=3,
        used_estimators=[VirtualTwinRF]
    ):
        """
        Perform the repeated KFold cross-validation and returs the results.

        Parameters
        ----------
        data : pd.DataFrame
            The whole dataset.
        treatment_names : list of strings
            What treatments to cross validate on. The datasets contains
            treatments 'treat_1' to 'treat_6', e.g.
            ['treat_1', 'treat_2', 'treat_3']
        repetitions : int, optional
            The number of repetitions for repeated KFold. The default is 1.
        folds : int, optional
            The number of folds for repeated KFold. The default is 3.
        used_estimators : list of HTEestimators, optional
            DESCRIPTION. The default is [VirtualTwinRF].

        Returns
        -------
        Results : MisraMatching.CV_Results object
            Contains the history and summaries of the performed repeated
            KFold Crossvalidation. See in its docstring for more information.

        """
        print(
            f"Starting Repeated KFold Misra Matching with\
              \n{repetitions} repetitions of {folds} folds."
        )
        Results = MisraMatching.CV_Results(folds=folds, reps=repetitions)
        fold_iterator = RepeatedKFold(
            n_splits=folds, n_repeats=repetitions, random_state=42
        )
        for train, test in fold_iterator.split(data):
            data_train = data.iloc[train].reset_index(drop=True)
            data_test = data.iloc[test].reset_index(drop=True)

            Shrinker = ShrinkageEstimators(data_test)

            summaries, assignments = zip(
                *[
                    self.getStatsForModel(
                        ModelClass,
                        data_train,
                        data_test,
                        treatment_names,  Shrinker,
                        return_assignments=True,
                    )
                    for ModelClass in used_estimators
                ]
            )
            ates = self.getATEs(data_test, used_treatments=treatment_names)

            Results.appendFoldStats(summaries, ates)
            Results.appendAssignments(assignments, test)
            Results.updateIndex()
        Results.getStatistics()
        return Results

    class CV_Results:
        """
        Generates and contains the results from the repeated KFold cross-
        validation.

        Relevant attributes
        ----------
        rep_stats: list of pd.DataFrames
            Summary statistics (means of matched outcomes, number of matched
            outcomes for each estimator as well as overall) for each
            repetition and all respective folds.

        overall_stats: pd.DataFrame
            Summary statistics (means of matched outcomes, number of matched
            outcomes for each estimator as well as overall) calculated over
            all repetitions and all folds.

        history: list of pd.DataFrames


        """

        current_rep = 0
        current_fold = 0

        def __init__(self, folds: int, reps: int):
            """


            Parameters
            ----------
            folds : int
                DESCRIPTION.
            reps : int
                DESCRIPTION.

            """
            self.total_folds = folds
            self.total_reps = reps
            self.summarystats = [
                [None for f in range(folds)] for r in range(reps)]
            self.history = [pd.DataFrame() for r in range(reps)]

        def appendFoldStats(self, dataframes, ates):
            """
            Append the sum of matched outcomes for the estimators and the
            overall sum to the results object.

            Parameters
            ----------
            dataframes : list of pd.DataFrames
                Contains the summary dataframes for the respective estimators.
            ates : pd.DataFrame
                Contains the summary dataframe for the overall outcomes.

            """
            merged_model_summaries = pd.concat(dataframes)
            self.summarystats[self.current_rep][self.current_fold] = pd.concat(
                [merged_model_summaries, ates]
            )
            return self

        def appendAssignments(self, assignments, indices):
            """
            Appends the predicted optimal treatments to the data storage.

            Parameters
            ----------
            assignments : pd.Series
                Predicted otpimal treatments (as str).
            indices: list
                Indices of the respective slice of the overall dataframe (
                most likely the indices of the current test data set).
            """
            df = pd.concat(assignments, axis=1)
            df.set_index(indices, inplace=True)
            self.history[self.current_rep] = pd.concat(
                [self.history[self.current_rep], df]
            )
            return self

        def updateIndex(self):
            """Updates the index pointers"""
            print(
                f"Finished Fold {self.current_fold+1} of repetition {self.current_rep+1}"
            )
            if self.current_fold < self.total_folds - 1:
                self.current_fold += 1
            elif self.current_rep < self.total_reps - 1:
                self.current_fold = 0
                self.current_rep += 1
            else:
                self.current_fold = "CV finished"
                self.current_rep = "CV finished"
            return self

        def getStatistics(self):
            """
            Generate the mean matched outcomes for each repetition as well
            as for the over repeated kfold cross validation.
            """
            rep_sums = [sum(foldsum) for foldsum in self.summarystats]
            overall_sum = sum(rep_sums)

            self.rep_stats = [self.getMeans(repsum) for repsum in rep_sums]
            self.overall_stats = self.getMeans(overall_sum)
            pass

        @staticmethod
        def getMeans(df):
            """
            From the summary dataframe of sums, convert to means. Divide
            the Outcome (Y) rows by the corresponding Count (N) rows.

            """
            for z in range(0, int(len(df) / 2)):
                df.iloc[2 * z] = df.iloc[2 * z] / df.iloc[2 * z + 1]
            return df

    @staticmethod
    def getMatchingPredictions(df, tau_pred, model_name: str):
        """
        Get the sum of outcomes for the Misra-Matched predictions.

        Parameters
        ----------
        df : pd.DataFrame
            Contains the (test-)data with actual (randomized) treatment
            assignments and respective observed outcomes.
        tau_pred : pd.DataFrame
            Contains the predicted individual treatment effects for each
            used treatment. Has to match the rows of the df (test-)data.
        model_name : str
            Model name to append to row names.

        Returns
        -------
        summary_df : pd.DataFrame
            Contains two rows, sum of matched outcomes and sum of counts of
            matched outcomes. Columns are treatments.

        """
        y_list = []
        n_list = []
        used_treatments = list(tau_pred.columns)
        for treatment in used_treatments:
            df_matched_assignment = df[
                (tau_pred.idxmax(axis=1) == treatment) & (df[treatment] == 1)
            ]
            y_list.append(df_matched_assignment.buttonpresses.sum())
            n_list.append(int(len(df_matched_assignment)))

        y_list.append(sum(y_list))
        n_list.append(sum(n_list))

        summary_df = pd.DataFrame.from_dict(
            {f"{model_name}Y": y_list, f"{model_name}N": n_list},
            orient="index",
            columns=used_treatments + ["overall"],
        )
        return summary_df

    @staticmethod
    def getATEs(data, used_treatments):
        """
        Get the sum of outcomes for each treatment as well as over all treatments
        to later compute the average treatment effects.

        Parameters
        ----------
        data : pd.DataFrame
            Contains the data including X, Y and T.

        Returns
        -------
        summary_df : pd.DataFrame
            Contains the sum of the outcomes of all treatments and for all
            treatments and the respective number of observations.
        """
        y_list = [
            data[data[treatment] == 1].buttonpresses.sum()
            for treatment in used_treatments
        ]
        y_list.append(sum(y_list))

        n_list = [int(len(data[data[treatment] == 1]))
                  for treatment in used_treatments]
        n_list.append(sum(n_list))

        summary_df = pd.DataFrame.from_dict(
            {"AllY": y_list, "AllN": n_list},
            orient="index",
            columns=list(used_treatments) + ["overall"],
        )
        return summary_df





if __name__ == "__main__":
    pass
  
    
    
    

    
    
    
