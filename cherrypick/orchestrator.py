
import pandas as pd
import time
import shap
import numpy as np
import joblib
import logging as log
import matplotlib.pyplot as plt
from typing import Literal, List, Dict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import box
from rich.table import Table
from rich.console import Console
from sklearn import tree
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.decomposition import PCA, TruncatedSVD
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (
                            RandomForestClassifier, RandomForestRegressor,
                            AdaBoostClassifier, AdaBoostRegressor,
                            GradientBoostingClassifier, GradientBoostingRegressor
                            )
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, r2_score, mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

class Orchestrator:

    """
    Orchestrator class to automate model training, evaluation,
    and selection for regression and classification tasks.

    Parameters
    ----------
    problem_statement : str
        Type of problem based on the dataset. Must be either
        ``'regression'`` or ``'classification'``.

    focus_regressor : str, default='mse'
        Metric used for selecting the best regression estimator.

        - `'mse'` - Mean Squared Error
        - `'mae'` - Mean Absolute Error
        - `'rmse'` - Root Mean Squared Error

    focus_classifier : str, default='f1score'
        Metric used for selecting the best classification estimator.

        - ``'recall'`` - Recall score
        - ``'precision'`` - Precision score
        - ``'f1score'`` - F1 score

    train : tuple
        Training data in the tuple format ``(X_train, y_train)``.

    test : tuple
        Testing data in the tuple format ``(X_test, y_test)``.

    file_dir : str
        Directory where the best estimator will be saved.
        For example, if a folder ``model/`` exists, use ``file_dir='model'``.
    
    Examples
    ---------
    >>> orch = Orchestrator(
                train = train,
                test=test,
                problem_statement='classification', ## for classification
                focus_classifier='f1score',
                file_dir='model'
            )
            
    >>> orch = Orchestrator(
                train = train,
                test=test,
                problem_statement='regression',
                focus_regressor='mae',
                file_dir='model'
            )
    """

    def __init__(self, train: tuple[pd.DataFrame, pd.Series], test:tuple[pd.DataFrame, pd.Series], file_dir:str, problem_statement : Literal['regression', 'classification'], seed:int = 42, focus_classifier: Literal['recall', 'precision', 'f1score'] = 'f1score', focus_regressor:Literal['mse', 'mae', 'rmse'] = 'mse' ):
        self.X_train, self.y_train = train
        self.X_test, self.y_test = test
        self.seed = seed
        self.file_dir = file_dir
        self.problem_statement = problem_statement
        self.focus_classifier = focus_classifier
        self.focus_regressor = focus_regressor

        ##Model data and persistence 
        self.best_model = []
        self.model_data = {}
        self.model_data_classify = {}
        self.SCALER_CONFIG = {}
    
    @property
    def best_estimator(self):
        """
    Returns best performing trained model.

    Code
    --------
    >>> orch.best_estimator

    Returns
    -------
    object
        The estimator with the highest performance based on
        the selected evaluation metric.
        """
        if self.best_model[0]:
            return self.best_model[0]
        else:
            raise ValueError("No model trained yet, use orchestrate(problem_statement : str , focus_classifier: str = 'f1score', focus_regressor:str = 'mse') to access best performing model")

    def __adjusted_r2score(self, r2):
        return 1 - ((1-r2) * (1-len(self.X_train)) / len(self.X_train) - (self.X_train.shape[1]) - 1)  

    def orchestrate(self):
        '''
        The function `orchestrate()` triggers the ML-model orchestration by cherry-picking the best estimator.

        Code
        --------
        >>> orch.orchestrate() ## Orchestrates entire model training and selects best model based upon Orchestrator() configs.
        '''

        regressor_models = {
            "LinearRegression" : LinearRegression(),
            "RandomForestRegressor" : RandomForestRegressor(),
            "XgBoostRegressor" : XGBRegressor(),
            "SVR" : SVR(),
            "KNeighborsRegressor" : KNeighborsRegressor(),
            "DecisionTreeRegressor" : DecisionTreeRegressor(),
            "AdaboostRegressor": AdaBoostRegressor(),
            "GradientBoostRegressor" : GradientBoostingRegressor()
        }

        classification_models = dict(

            LogisticRegression = LogisticRegression(),
            SVC = SVC(),
            KNeighborsClassifier = KNeighborsClassifier(),
            RandomForestClassifier = RandomForestClassifier(),
            DecisionTreeClassifier = DecisionTreeClassifier(),
            XGBClassifier = XGBClassifier(),
            AdaBoostClassifier = AdaBoostClassifier(),
            GradientBoostingClassifier = GradientBoostingClassifier()

        )
        linear_model_regression = ['LinearRegression', "SVR", "KNeighborsRegressor"] 

        try:
            if self.problem_statement == 'regression':
                for name, model in regressor_models.items():
                    if name not in linear_model_regression:

                        ensemble = model.fit(self.X_train, self.y_train)
                        y_pred = ensemble.predict(self.X_test)

                        accuracy = r2_score(self.y_test, y_pred)
                        mae = mean_absolute_error(self.y_test, y_pred)
                        mse = mean_squared_error(self.y_test, y_pred)
                        rmse = (mse) ** 0.5
                        adj_r2 = self.__adjusted_r2score(r2=accuracy) ## will be soon applying Adjusted R2_score for the penalisation influence of the useless features.

                        result = dict(
                            estimator = ensemble,
                            accuracy = accuracy,
                            mse=mse,
                            mae=mae,
                            rmse = rmse
                        )
                        self.model_data[name] = result
                        

                    else:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(self.X_train)
                        X_test_scaled = scaler.transform(self.X_test)

                        regressor = model.fit(X_train_scaled, self.y_train)
                        y_pred = regressor.predict(X_test_scaled)

                        accuracy = r2_score(self.y_test, y_pred)
                        mae = mean_absolute_error(self.y_test, y_pred)
                        mse = mean_squared_error(self.y_test, y_pred)
                        rmse = (mse) ** 0.5
                        
                        result = dict(
                            estimator = regressor,
                            accuracy = accuracy,
                            mse=mse,
                            mae=mae,
                            rmse = rmse
                        )
                        model_name = type(regressor).__name__
                            
                        self.SCALER_CONFIG[model_name] = scaler
                        self.model_data[name] = result
                        # joblib.dump(scaler, f'{self.file_dir}/scaler.pkl')
                        

                if self.focus_regressor == "mse":
                    
                    data_metrics = pd.DataFrame(self.model_data).T.sort_values(by=['mse', 'accuracy'], ascending=[True, False])
                    best_model = data_metrics['estimator'].iloc[0]
                    model_name = type(best_model).__name__

                    if model_name in linear_model_regression:
                        scaler = self.SCALER_CONFIG[model_name]
                        joblib.dump(scaler,  f"{self.file_dir}/scaler_r.pkl")
                    
                   
                    self.best_model.append(best_model)
                    joblib.dump(best_model, f"{self.file_dir}/{model_name}.pkl")
                    print("-------------Demorgraphics-------------")
                    print(f"Best Model : {model_name}")
                    print(f"{model_name} Accuracy(R2 SCore) : {data_metrics['accuracy'].iloc[0]}")
                    print(f"{model_name} MSE : {data_metrics['mse'].iloc[0]}")
                    print("---------------------------------------")
                    return best_model


                elif self.focus_regressor == "mae":
                    data_metrics = pd.DataFrame(self.model_data).T.sort_values(by=['mae', 'accuracy'], ascending=[True, False])
                    best_model = data_metrics['estimator'].iloc[0]
                    model_name = type(best_model).__name__

                    if model_name in linear_model_regression:
                        scaler = self.SCALER_CONFIG[model_name]
                        joblib.dump(scaler,  f"{self.file_dir}/scaler_r.pkl")
                   
                    self.best_model.append(best_model)
                    joblib.dump(best_model, f"{self.file_dir}/{model_name}.pkl")
                    print("-------------Demorgraphics-------------")
                    print(f"Best Model : {model_name}")
                    print(f"{model_name} Accuracy(R2 SCore) : {data_metrics['accuracy'].iloc[0]}")
                    print(f"{model_name} MAE : {data_metrics['mae'].iloc[0]}")
                    print("---------------------------------------")
                    return best_model

                elif self.focus_regressor == "rmse":
                    
                    data_metrics = pd.DataFrame(self.model_data).T.sort_values(by=['rmse', 'accuracy'], ascending=[True, False])
                    best_model = data_metrics['estimator'].iloc[0]
                    model_name = type(best_model).__name__

                    if model_name in linear_model_regression:
                        scaler = self.SCALER_CONFIG[model_name]
                        joblib.dump(scaler,  f"{self.file_dir}/scaler_r.pkl")
                   
                    self.best_model.append(best_model)
                    joblib.dump(best_model, f"{self.file_dir}/{model_name}.pkl")
                    print("-------------Demorgraphics-------------")
                    print(f"Best Model : {model_name}")
                    print(f"{model_name} Accuracy(R2 SCore) : {data_metrics['accuracy'].iloc[0]}")
                    print(f"{model_name} RMSE : {data_metrics['rmse'].iloc[0]}")
                    print("---------------------------------------")
                    return best_model

            if self.problem_statement == 'classification':
                
                for name, model in classification_models.items():
                    if name not in ["LogisticRegression", "SVC", "KNeighborsClassifier"]:

                            estimator = model.fit(self.X_train, self.y_train)
                            y_pred = estimator.predict(self.X_test)

                            accuracyscore = accuracy_score(self.y_test, y_pred)
                            precisionscore = precision_score(self.y_test, y_pred, average='weighted')
                            recallscore = recall_score(self.y_test, y_pred, average='weighted')
                            f1scores = f1_score(self.y_test, y_pred, average='weighted')

                            result = dict(
                                estimator = estimator,
                                accuracy = accuracyscore,
                                precision = precisionscore,
                                recall = recallscore,
                                f1score = f1scores
                            )
                            self.model_data_classify[name] = result
                            

                    else:
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(self.X_train)
                            X_test_scaled = scaler.transform(self.X_test)

                            estimator = model.fit(X_train_scaled, self.y_train)
                            y_pred = estimator.predict(X_test_scaled)

                            accuracyscore = accuracy_score(self.y_test, y_pred)
                            precisionscore = precision_score(self.y_test, y_pred, average='weighted')
                            recallscore = recall_score(self.y_test, y_pred, average='weighted')
                            f1scores = f1_score(self.y_test, y_pred, average='weighted')

                            result = dict(
                                estimator = estimator,
                                accuracy = accuracyscore,
                                precision = precisionscore,
                                recall = recallscore,
                                f1score = f1scores
                            )

                            model_name = type(estimator).__name__
                            
                            self.SCALER_CONFIG[model_name] = scaler
                            self.model_data_classify[name] = result
                            # joblib.dump(scaler, f'{self.file_dir}/scaler.pkl')
                            
                try:
                    if self.focus_classifier == 'precision':
                            
                            data_metrics = pd.DataFrame(self.model_data_classify).T.sort_values(by='precision', ascending=False)
                            best_model = data_metrics['estimator'].iloc[0]
                            model_name = type(best_model).__name__

                            if model_name in ["LogisticRegression", "SVC", "KNeighborsClassifier"]:
                                scaler = self.SCALER_CONFIG[model_name]
                                joblib.dump(scaler,  f"{self.file_dir}/scaler_c.pkl")
                        
                            self.best_model.append(best_model)
                            joblib.dump(best_model, f"{self.file_dir}/{model_name}.pkl")
                            print("-------------Demographics-------------")
                            print(f"Best Model : {model_name}({self.focus_classifier})")
                            print(f"{model_name} Accuracy : {data_metrics['accuracy'].iloc[0]}")
                            print(f"{model_name} Precision : {data_metrics['precision'].iloc[0]}")
                            print(f"{model_name} Recall : {data_metrics['recall'].iloc[0]}")
                            print(f"{model_name} f1score : {data_metrics['f1score'].iloc[0]}")
                            print("---------------------------------------")
                            return best_model
                            
                        
                    elif self.focus_classifier == 'recall':
                            data_metrics = pd.DataFrame(self.model_data_classify).T.sort_values(by='recall', ascending=False)
                            best_model = data_metrics['estimator'].iloc[0]
                            model_name = type(best_model).__name__

                            if model_name in ["LogisticRegression", "SVC", "KNeighborsClassifier"]:
                                scaler = self.SCALER_CONFIG[model_name]
                                joblib.dump(scaler,  f"{self.file_dir}/scaler_c.pkl")
                        
                            self.best_model.append(best_model)
                            joblib.dump(best_model, f"{self.file_dir}/{model_name}.pkl")
                            print("-------------Demographics-------------")
                            print(f"Best Model : {model_name}({self.focus_classifier})")
                            print(f"{model_name} Accuracy : {data_metrics['accuracy'].iloc[0]}")
                            print(f"{model_name} Precision : {data_metrics['precision'].iloc[0]}")
                            print(f"{model_name} Recall : {data_metrics['recall'].iloc[0]}")
                            print(f"{model_name} f1score : {data_metrics['f1score'].iloc[0]}")
                            print("---------------------------------------")
                            return best_model
                        
                    elif self.focus_classifier == 'f1score':
                            data_metrics = pd.DataFrame(self.model_data_classify).T.sort_values(by='f1score', ascending=False)
                            best_model = data_metrics['estimator'].iloc[0]
                            model_name = type(best_model).__name__

                            if model_name in ["LogisticRegression", "SVC", "KNeighborsClassifier"]:
                                scaler = self.SCALER_CONFIG[model_name]
                                joblib.dump(scaler,  f"{self.file_dir}/scaler_c.pkl")
                        
                            self.best_model.append(best_model)
                            joblib.dump(best_model, f"{self.file_dir}/{model_name}.pkl")
                            print("-------------Demographics-------------")
                            print(f"Best Model : {model_name}({self.focus_classifier})")
                            print(f"{model_name} Accuracy : {data_metrics['accuracy'].iloc[0]}")
                            print(f"{model_name} Precision : {data_metrics['precision'].iloc[0]}")
                            print(f"{model_name} Recall : {data_metrics['recall'].iloc[0]}")
                            print(f"{model_name} f1score : {data_metrics['f1score'].iloc[0]}")
                            print("---------------------------------------")
                            return best_model
                        
                    else:
                        print("Invalid Metric Please consider Recall, Precision or F1_Score")
                except Exception as error:
                    print(f"{error}")

        except Exception as error:
            print(error)
                

    def cv(self, type_cv:str, param_grid : dict, scoring_type:str, n_jobs:int=-1, cv: int=5):
        '''
        Under **progress** will be available soon!
        '''
        try:
            if type_cv == 'randomised':
                random = RandomizedSearchCV(
                     estimator=self.best_model[0],
                     cv=cv,
                     param_distributions=param_grid,
                     scoring=scoring_type,
                     n_jobs=n_jobs,
                     refit=True
                    )
                
                random_cv_model = random.fit(self.X_train, self.y_train)
                best_param = random_cv_model.best_params_
                best_score = random_cv_model.best_score_
                best_model = random.best_estimator_

                result = dict(
                     best_params = best_param,
                     best_scores = best_score,
                     best_model = best_model                    
                )
                for keys, values in result.items():
                      print(f'{keys} : {values}')
                joblib.dump(result['best_model'], f"{self.file_dir}/{result['best_model']}.pkl")
                
                return result['best_model']


            elif type_cv == 'gridsearch':
                gridcv = GridSearchCV(
                      estimator=self.best_model[0],
                      cv = cv,
                      param_grid=param_grid,
                      scoring=scoring_type,
                      n_jobs=n_jobs,
                      refit=True
                 )
                gridcv_model = gridcv.fit(self.X_train, self.y_train)
                best_param = gridcv_model.best_params_
                best_score = gridcv_model.best_score_
                best_model = gridcv_model.best_estimator_

                result = dict(
                     best_params = best_param,
                     best_scores = best_score,
                     best_model = best_model                                        
                )
                for keys, values in result.items():
                      print(f'{keys} : {values}')
                joblib.dump(result['best_model'], f"{self.file_dir}/{result['best_model']}.pkl")

                return result['best_model']

            else:
                 print("Please enter the correct Cross Validation type") 


        except Exception as error:
             print(error)


    def critique(self, cv:int = 5, scoring:str='neg_mean_squared_error', topkmodel:int = None) -> str:
        """
        Evaluate model generalization and diagnose overfitting or underfitting
        (bias-variance tradeoff) using cross-validation.

        Parameters
        ----------
        cv : int, default=5
            Number of cross-validation folds.

        scoring : str
            Metric used to evaluate each cross-validation fold.

        topkmodel : int, optional
            Index of the model to evaluate from the top-k selected models.
            If not provided, the best estimator is used by default.

        Returns
        -------
        str
            Alert message including the relative gap:

            .. math::

                \\text{Relative Gap} = \\frac{\\text{overfitting\\_gap}}{\\text{MSE (training)}}

            Where:

            - **overfitting_gap** = mean cross-validation score − training MSE

        Notes
        -----
        - Helps identify whether the model is overfitting or underfitting.
        - Higher variance indicates overfitting.
        - High bias indicates underfitting.

        Code
        --------
        >>> orch.critique(cv=n, scoring='neg_mean_squared_error') ## Checks the sanity(bias-variance tradeoffs) for best model
        >>> orch.critique(cv=n, scoring='neg_mean_squared_error', topkmodel = model) ## Checks the sanity(bias-variance tradeoffs) for custom model

        """

        try:
            scores = cross_val_score(self.best_model[0], cv=cv, X=self.X_train, y=self.y_train, scoring=scoring)
            mean_score_cv =  -np.mean(scores)

            y_train_pred = self.best_model[0].predict(self.X_train)
            y_test_pred = self.best_model[0].predict(self.X_test)
            
            if scoring == "neg_mean_squared_error":

                ## Scores based upon
                mse_train = mean_squared_error(self.y_train, y_train_pred)

                ## Overfitting gap calculation
                overfitting_gap = mean_score_cv - mse_train
                relative_gap = overfitting_gap/mse_train

                if relative_gap < 0.05:
                    message =  "NO Overfitting"
                elif relative_gap < 0.15:
                    message =  "Mild Overfitting"
                elif relative_gap < 0.30:
                    message =  "HIGH Overfitting"
                elif relative_gap < 0:
                    message =  "Randomness"

                else:
                    pass
            
            if scoring == 'r2':
                r2_train = r2_score(self.y_train, y_train_pred)

                overfitting_gap = mean_score_cv - r2_train
                relative_gap = overfitting_gap / r2_train

                if relative_gap < 0.05:
                    message =  "NO Overfitting"
                elif relative_gap < 0.15:
                    message = "Mild Overfitting"
                elif relative_gap < 0.30:
                    message =  "HIGH Overfitting"
                elif relative_gap < 0:
                    message= "Randomness"

                else:
                    pass

            ALERT_CONFIG = {
                "NO Overfitting" : "With Relative Overfitting Gap = {0}, good to go for predictions!".format(relative_gap),
                "Mild Overfitting" : "With Relative Overfitting Gap = {0}, might need to train the model appropriately".format(relative_gap),
                "HIGH Overfitting" : "With Relative Overfitting Gap = {0}, high chances of data leakage or dataset is too simple".format(relative_gap),
                "Randomness" : "With Relative Overfitting Gap = {0}, might be data leakage or model randomness".format(relative_gap)
            }

            return print(ALERT_CONFIG[message])
                 
        except Exception as err:
             print(err)
            
            

    def topkmodel(self, access_estimator: int = None, threshold: float | int | None = None) -> pd.DataFrame | None:

        """
        Retrieve top-performing models or access a specific estimator.

        Parameters
        ----------
        access_estimator : int, optional
            Index of the estimator to access from the ranked model list
            (1st, 2nd, ..., nth). If provided, returns the selected estimator.

        threshold : float or int, optional
            Threshold value used to filter models based on the evaluation metric.
            Only models meeting the threshold criteria are returned.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame containing ranked models and their evaluation metrics.

            If ``access_estimator`` is provided, returns the selected estimator
            instead of a DataFrame.

        Notes
        -----
        - Models are ranked based on the selected evaluation metric.
        - If both parameters are ``None``, the full ranked model table is returned.
        - If ``access_estimator`` is provided, threshold filtering is ignored.

        Code
        --------
        >>> orch.topkmodel() ## returns leaderboard of top K models
        >>> orch.topkmodel(access_estimator = n) ## returns choosen model from nth rank(1st - nth)
        """
        
        console = Console()

        if self.problem_statement == 'regression':
            data = self.model_data
            df = pd.DataFrame(data).T

            if self.focus_regressor == "mse":
                df = df.sort_values(by=['mse', 'accuracy'], ascending=[True, False])
                metric = 'mse'

            elif self.focus_regressor == "mae":
                df = df.sort_values(by=['mae', 'accuracy'], ascending=[True, False])
                metric = 'mae'

            elif self.focus_regressor == "rmse":
                df = df.sort_values(by=['rmse', 'accuracy'], ascending=[True, False])
                metric = 'rmse'

        else:
            data = self.model_data_classify
            df = pd.DataFrame(data).T

            if self.focus_classifier == 'precision':
                df = df.sort_values(by='precision', ascending=False)
                metric = 'precision'

            elif self.focus_classifier == 'recall':
                df = df.sort_values(by='recall', ascending=False)
                metric = 'recall'

            elif self.focus_classifier == 'f1score':
                df = df.sort_values(by='f1score', ascending=False)
                metric = 'f1score'

        # ranking
        df = df.reset_index(drop=True)
        df.index = range(1, len(df) + 1)
        df.insert(0, "Rank", df.index)

        # threshold logic
        if threshold is not None:
            if self.problem_statement == 'regression':
                df = df[df[metric] <= threshold]   # lower is better
            else:
                df = df[df[metric] >= threshold]   # higher is better

        df = df.round(4)

       
        table = Table(
            title="  Model Arena",
            box=box.DOUBLE_EDGE,
            # box=box.SQUARE_DOUBLE_HEAD,
            show_lines=True,
            header_style="bold white",
            border_style="bright_blue"
            
        )

        for col in df.columns:
            style = "bright_cyan" if col == "estimator" else "white"
            table.add_column(col, justify="center", style=style)

        for _, row in df.iterrows():

            values = []
            row_style = ""

            for col in df.columns:
                val = row[col]

                if col == "Rank":
                    if val == 1:
                        val = "[bold yellow]🥇1[/bold yellow]"
                        row_style = "bold yellow"
                    elif val == 2:
                        val = "[bold #C0C0C0]🥈2[/bold #C0C0C0]"
                        row_style = "#C0C0C0"
                    elif val == 3:
                        val = "[bold #CD7F32]🥉3[/bold #CD7F32]"
                        row_style = "#CD7F32"
                    else:
                        val = f"[white]{val}[/white]"

                elif isinstance(val, (int, float)):
                    if val >= 0.9:
                        val = f"[bright_green]{val}[/bright_green]"
                    elif val >= 0.75:
                        val = f"[yellow]{val}[/yellow]"
                    else:
                        val = f"[red]{val}[/red]"

                values.append(str(val))

            table.add_row(*values, style=row_style)

        panel = Panel.fit(
            table,
            title="[bold bright_cyan] Top Models",
            border_style="bright_magenta"
        )

        console.print(panel)

        if access_estimator is not None:
            return df['estimator'].iloc[access_estimator - 1]
        

    def auto_explain(self, n_classes:int = None, size:tuple|None = None, model : str = 'best'):
        
        """
        Generate SHAP-based explanations for trained models.

        Provides automatic model interpretability using SHAP
        (SHapley Additive Explanations) for both regression and
        classification tasks. Supports TreeExplainer and LinearExplainer,
        along with visualization tools such as summary plots and bar plots.

        Parameters
        ----------
        n_classes : int
            Number of unique output classes. Required for classification tasks.

        size : tuple, optional
            Figure size used for resizing Decision Tree visualizations.

        model : str, default='best'
            Specifies which model to explain.

            - ``'best'`` : Uses the top-performing estimator.

        Returns
        -------
        None
            Generates SHAP visualizations such as summary plots and bar plots.
            For classification tasks, ``n_classes`` must be specified.

        Notes
        -----
        - Uses SHAP TreeExplainer for tree-based models.
        - Uses SHAP LinearExplainer for linear models.
        - Supports both regression and classification workflows.
        - Outputs include summary plots and feature importance visualizations.

        Code
        --------
        >>> orch.auto_explain() ## for best model explaination
        >>> orch.auto_explain(n_classes=class_ids) ## classification type best model explaination
        >>> orch.auto_explain(model=model) ## custom model based explaination 
        """
    
        
        ensembleClassifier_tuple = (
                                    RandomForestClassifier,
                                    AdaBoostClassifier, 
                                    XGBClassifier, 
                                    GradientBoostingClassifier
                                )
        
        linearClassifier_tuple = (
                                  LogisticRegression, 
                                )
        
        dtree_classifier = DecisionTreeClassifier

        ensembleRegressor_tuple = (
                                   RandomForestRegressor, 
                                   AdaBoostRegressor, 
                                   XGBRegressor, 
                                   GradientBoostingRegressor
                                   )
 

        if model != 'best':
            model = model

        elif model == 'best': 
            model = self.best_model[0]
        else:
            raise ValueError("Model is either not supported or just there is no model that exists")
        

        if self.problem_statement == 'classification':

            if n_classes is not None and isinstance(n_classes,  float):
                raise ValueError("Provide the number of unique classes in Integer format")
            
            if n_classes != len(self.y_train.value_counts()):
                raise ValueError("n_classes is either less than or greater than number of classes make sure its exactly equal to n_classes")
            

            
            if isinstance(model, ensembleClassifier_tuple):
                explainer = shap.TreeExplainer(model, self.X_train)

                shap_values_train = explainer(self.X_train, check_additivity=False)
                shap_values_test = explainer(self.X_test, check_additivity=False)

                if n_classes is not None:
                            
                    for class_id in range(n_classes): 
                        plt.figure()   
                        plt.title(f"For class_id {class_id}")
                        shap.plots.bar(shap_values_test[..., class_id])
                        

                    if(
                        self.X_train is not None and self.X_test is not None and 

                        not self.X_train.empty and not self.X_test.empty
                    ):

                        plt.title("Summary Plot for X_train")
                        shap.summary_plot(shap_values_train, self.X_train)
                    

                        
                        plt.title("Summary Plot for X_test")
                        shap.summary_plot(shap_values_test, self.X_test)
                        


                elif n_classes is None:
                    print("bar successful")
                    plt.title("Global Features Importance")
                    shap.plots.bar(shap_values_test)

                else:
                    raise ValueError("X_train or X_test might be None or invalid")
            

            elif isinstance(model, dtree_classifier):
                    plt.figure(figsize=size)
                    tree = tree.plot_tree(self.best_model, filled=True, feature_names=self.X_train.columns, class_names=True)

            
            elif isinstance(model, linearClassifier_tuple):

                explainer = shap.LinearExplainer(model, self.X_train)
                
                shap_values_train = explainer(self.X_train)
                shap_values_test = explainer(self.X_test)

                if n_classes is not None:
                    if len(shap_values_test.shape) >= 3:
                        for class_id in range(0, n_classes):
                            plt.title(f"Class {class_id}")
                            shap.plots.bar(shap_values_test[..., class_id])

                    if(
                        self.X_train is not None and self.X_test is not None and 

                        not self.X_train.empty and not self.X_test.empty
                    ):

                        plt.title("Summary Plot for X_train (Classification)")
                        shap.summary_plot(shap_values_train, self.X_train)
                    
                        plt.title("Summary Plot for X_test (Classification)")
                        shap.summary_plot(shap_values_test, self.X_test) 
                
                elif n_classes is None and len(shap_values_test.shape) == 2:
                    plt.title("Global Feature Imortance (BAR)")
                    shap.plots.bar(shap_values_test)

        if self.problem_statement == 'regression':
            if isinstance(model, ensembleRegressor_tuple):

                explainer = shap.TreeExplainer(model, self.X_train)
                shap_values_test = explainer(self.X_test, check_additivity=False)  
                shap_values_train = explainer(self.X_train, check_additivity=False)
                
                plt.title("Summary Plot for X_train (Regression)")
                shap.summary_plot(shap_values_train, self.X_train)

                plt.title("Summary Plot for X_test (Regression)")
                shap.summary_plot(shap_values_test, self.X_test)

        else:
            None


    
      
    
    

        

                

                    
