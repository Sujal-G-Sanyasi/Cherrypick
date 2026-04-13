import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from typing import Literal

import warnings as war
war.filterwarnings('ignore')

class OutlierPruner:
    """
    OutlierPruner provides statistical and ML-based methods
    for detecting and removing outliers from a dataset.

    Parameters
    ----------
    method : {'iqr', 'zscore', 'mod_zscore', 'isoforest', 'lof'}
        Method used for outlier detection.

        - ``iqr`` : Interquartile Range method
        - ``zscore`` : Standard Z-score normalization
        - ``mod_zscore`` : Modified Z-score

        .. math::

            Z = 0.6745 * (X - median) / MAD

        Where:

        - **median** : median of the sample data
        - **MAD** : median absolute deviation
        - **X** : sample data points

        - ``isoforest`` : Isolation Forest (ensemble-based anomaly detection)
        - ``lof`` : Local Outlier Factor (density-based detection)

    df : pandas.DataFrame
        Input dataset on which outlier pruning will be applied.

    col : str
        Column name used for outlier detection in statistical methods.

    Notes
    -----
    - Statistical methods require a specific column (``col``).
    - ML-based methods (Isolation Forest, Local Outlier Factor) operate on numerical features.
    - Modified Z-score is robust to extreme values as it uses the median instead of the mean.

    Code
    -----
    >>> pruner = OutlierPruner(df=df, method='isoforest', col=column)
     
    >>> pruner.remove_outlier() ## removes the Outliers using Isolation forest 
    """

    def __init__(self, method: Literal['iqr', 'zscore', 'mod_zscore', 'isoforest', 'lof'], df:pd.DataFrame, col:str):
        self.df = df
        self.col = col
        self.method = method
        
    
    def __iqr(self):
            
                Q1 = self.df[self.col].quantile(0.25)
                Q3 = self.df[self.col].quantile(0.75)
                IQR = Q3 - Q1

                lower_fence = Q1 - 1.5 * IQR
                upper_fence = Q3 + 1.5 * IQR

                return  self.df[(self.df[self.col] >= lower_fence) & (self.df[self.col] <= upper_fence)]   
            
    def __zscore(self):
            z = zscore(self.df[self.col])
            return self.df[np.abs(z) < 3]

    def __isoforest(self):
            isolate = IsolationForest(contamination=0.3, n_jobs=-1, random_state=42)
            
            X = self.df.select_dtypes(include=np.number)
            labels_ = isolate.fit_predict(X)

            # outliers = np.where(labels_ == -1)[0]
            return self.df.iloc[labels_!= -1]
    
    def __lof(self):
            lof = LocalOutlierFactor(n_jobs=-1, n_neighbors=20, algorithm='kd_tree')
            X = self.df.select_dtypes(include = np.number)
            labels = lof.fit_predict(X)

            return self.df.iloc[labels != -1]
            

    def __modded_zscore(self):
        
        df1=self.df
        median = np.median(df1[self.col])
        mad = np.median(np.abs(df1[self.col] - median))
        ## If MAD value == 0, then it will return original DataFrame instead of garbage value and prevent division by zero error
        if mad  == 0 :
              return self.df
        
        mod_zscore = 0.6745 * (df1[self.col] - median)/mad

        normal_data = df1[mod_zscore.abs() < 3]
        outliers = df1[mod_zscore.abs() > 3]     
        
        return normal_data


    def remove_outlier(self):
        '''
        Calling this function will transform dataset with configuration provided to **OutlierPruner**. 
        '''
        try:                
            METHOD_CONFIG = { 
                "iqr" : self.__iqr,
                "zscore" : self.__zscore,
                "mod_zscore":self.__modded_zscore,
                "isoforest" : self.__isoforest,
                "lof" : self.__lof
            }

            return METHOD_CONFIG[self.method]()
            
        except KeyError:
            raise ValueError(f"Provide an appropriate method : {self.method}") 
        except Exception as err:
              raise ValueError(err)