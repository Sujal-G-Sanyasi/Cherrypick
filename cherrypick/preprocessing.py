import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from typing import Literal
import joblib
import seaborn as sns

import warnings as war
war.filterwarnings('ignore')

class Preprocessor:
    """
    Split dataset into training and testing sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing features and target variable.

    duplicate : str
        Removes duplicate values row-wise if ``duplicate = 'drop'``, else returns original value.

    Examples
    --------
    >>> preprocess = Preprocessor(df=df) ## Initialization of Preprocessor
    """
    def __init__(self, df, duplicate : Literal["drop"] = None): 
        
        if duplicate == "drop":
            self.df = df.drop_duplicates()
        else:
            self.df = df
    
    def fill_null(self, type:Literal['mean', 'median', 'mode'], columns:str) -> pd.DataFrame:
        """
        Method consisting several Imputers for handling Null values in a given dataset.

        Parameters
        ----------
        columns : str
            Desired column name of a input variable for imputer processing.

        type : {'mean', 'median', 'mode'}
            determines the type of Imputation techniques.

            - ``mean`` - Takes ``mean of all samples`` in a feature and substitutes the *NaN* value.

            - ``median`` - Takes ``median of all samples`` in a feature and substitutes the *NaN* value, aggresive and robust to Outliers.

            - ``mode`` - Takes the ``most frequently occuring value of a sample`` in a feature and substitute the *NaN* value, used for categorical features.

        Examples
        --------
        >>> preprocess.fill_null(type='mean', columns=column) ## Fill nulls by taking mean of all values in specified column
        >>> preprocess.fill_null(type='median', columns=column) ## Fill nulls by taking median of all values in specified column
        >>> preprocess.fill_null(type='mode', columns=column) ## Fill nulls by taking mode of all values in specified categorical column

        >>> print(df.isna().sum()) ## returns 0 as all nulls are handeled using Imputers
        Returns
        -------
        pd.DataFrame
            Cleaned dataset with all *NaN* values handeled with mean, mode and median imputation statistical techniques.
        """

        
        try:
            IMPUTER_CONFIG = {
                    "mean" : self.df[columns].fillna(self.df[columns].mean()),
                    "median" : self.df[columns].fillna(self.df[columns].median()),
                    "mode" : self.df[columns].fillna(self.df[columns].mode()[0])
            }

            self.df[columns] = IMPUTER_CONFIG[type]
            return self.df
        
        except Exception as err:
            raise Exception(f"Choose correct type and columns : {err}")

    def collinear(self, threshold:float, show:Literal[True, False] = False, method : Literal['spearman', 'pearson'] = 'pearson'):
        """
        Handle multicollinearity by identifying highly correlated features.

        This method detects correlated features based on a specified correlation
        threshold and statistical method.

        Parameters
        ----------
        threshold : float
            Correlation threshold above which features are considered collinear.

        method : {'spearman', 'pearson'}, default='pearson'
            Type of correlation method used:

            - ``spearman`` : Captures monotonic relationships using rank correlation
            - ``pearson`` : Measures linear correlation between features

        show : {True, False}, default=False
            Shows correlation matrix with list of collinear features if True, else just returns the list of collinear features
        
        Returns
        -------
        list
            List of correlated features recommended for removal.

        Notes
        -----
        - Helps reduce multicollinearity in datasets.
        - Improves model stability and interpretability.

        Examples
        --------
        >>> preprocess.collinear(threshold=0.85, type='pearson', show=true)
        """
        try:
            columns = set()
            correlation = self.df.corr(method = method)
            
            for i in range(len(correlation.columns)):
                for j in range(i):
                    if abs(correlation.iloc[i, j]) > threshold:
                        colname = correlation.columns[i]
                        columns.add(colname)
            
            if show == True:
                sns.heatmap(self.df.corr(), annot=True)           
                return sorted(columns)
            
            else:
                return sorted(columns)
        
        except Exception as error:
            print(f"Error on finding the Collinearity : {error}")

    def encoder(self, type: Literal['onehot', 'label'], train_data:tuple , test_data:tuple , column: str, encoder_dir : str):
        """
        Method consisting OneHotEncode and LabelEncoder for handling any Categorical feature in a given dataset and convert it into numeric data.

        Parameters
        ----------
        column : str
            Desired column name of a input variable for imputer processing.

        type : {'onehot', 'label'}
            determines the type of encoder to perform encodings.
                
                               - **Onehot** - Uses One Hot Encoding on a categorical column.
                                        **returns**
                                        Sparse matrix containing 1's and 0's
                              
                               - **label** - Uses LabelEncoder on target(output) column if there exist non-numeric categorical data.
                                        **returns**
                                        Matrix containing 1 to n, based upon the number of class
        
        train_data : tuple
            Arguement requires Training data i.e `X_train and y_train`.

        test_data : tuple
            Arguement requires Training data i.e `X_test and y_test`.

        encoder_dir : str
            File directory where the encoder needs to persist.


            
        Examples
        --------
        >>> preprocess.encoder(train_data = train, test_data = test, column = 'feature1', type = 'onehot', encoder_dir='encoder_') ## Onehot encodings for Input Features, returns Sparse matrix
        >>> preprocess.encoder(train_data = train, test_data = test, column = 'target', type = 'label', encoder_dir='encoder_') ## label encodings for Output/target feature

        Returns
        -------
        pd.DataFrame
            For **OneHotEncoder** returns encoded `X_train` and `X_test`.
            
            For **LabelEncoder** returns encoded `y_train` and `y_test`.

        """
        X_train, y_train = train_data
        X_test, y_test = test_data

        ENCODER_CONFIG = {
            "onehot": OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            "label": LabelEncoder()
        }

        encoder = ENCODER_CONFIG[type]

        if isinstance(encoder, OneHotEncoder):

            encoded_Xtrain = encoder.fit_transform(X_train[[column]])
            encoded_Xtest = encoder.transform(X_test[[column]])

            encoded_df_Xtrain = pd.DataFrame(
                encoded_Xtrain,
                columns=encoder.get_feature_names_out([column])
            ).reset_index(drop=True)

            encoded_df_Xtest = pd.DataFrame(
                encoded_Xtest,
                columns=encoder.get_feature_names_out([column])
            ).reset_index(drop=True)

            X_train = X_train.drop(columns=[column])
            X_test = X_test.drop(columns=[column])

            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            
            X_train = pd.concat([X_train, encoded_df_Xtrain], axis=1, ignore_index=False)
            X_test = pd.concat([X_test, encoded_df_Xtest], axis=1, ignore_index=False)

            joblib.dump(encoder, f"{encoder_dir}/onehot_encoder.pkl")

            return X_train, X_test

        elif isinstance(encoder, LabelEncoder):

            y_train = encoder.fit_transform(y_train)
            y_test = encoder.transform(y_test)
            
            joblib.dump(encoder, f"{encoder_dir}/label_encoder.pkl")

            return y_train, y_test

        else:
            raise ValueError("Invalid encoder type")  

        
    


        
