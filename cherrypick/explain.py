import shap
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import tree
from cherrypick.orchestrator import Orchestrator
from typing import Literal


def explainer(model, data, impact_type :Literal['pos', 'neg', 'all'] = 'all'):
    """
    Compute SHAP-based feature importance and return sorted impact values.

    This function uses SHAP's TreeExplainer to calculate feature contributions
    for a given model and dataset. It aggregates SHAP values across samples
    and (if applicable) across multiple classes, returning feature importance
    based on absolute SHAP magnitudes.

    Parameters
    ----------
    model : object
        A trained tree-based model compatible with shap.TreeExplainer
        (e.g., XGBoost, LightGBM, RandomForest).

    data : pandas.DataFrame
        Input dataset for which SHAP values are to be computed.
        Must contain only feature columns (no target column).

    impact_type : {'pos', 'neg', 'all'}, default='all'
        Type of feature impact to return:
        - '**pos**' - Returns features with positive contribution.
        - '**neg**' - Returns features with negative contribution.
        - '**all**' - Returns all features with overall importance
                  (absolute SHAP values).

    Returns
    -------
    result : pandas.DataFrame
        A sorted DataFrame containing feature importance:
        - For 'all' → columns: ['Features', 'Overall_Impact']
        - For 'pos' → columns: ['Features', 'Positive_Impact']
        - For 'neg' → columns: ['Features', 'Negative_Impact']

    shap_values : shap.Explanation
        Raw SHAP explanation object containing per-sample contributions.

    Notes
    -----
    - For multi-class models, SHAP values are averaged across classes.
    - Feature importance is computed using mean absolute SHAP values.
    - The function also stores SHAP values globally in `_shap_val`.

    Raises
    ------
    ValueError
        If `impact_type` is not one of {'pos', 'neg', 'all'}.

    Example
    -------
    >>> result, shap_vals = explainer(model, X_test, impact_type='all')
    >>> print(result.head())

    """

    ## All the Shap values with magnitude based as well!
    features = [ ]
    all_values = [ ]

    neg_values = [ ]
    neg_feature = [ ]

    pos_values = [ ]
    pos_feature = [ ]

    explain = shap.TreeExplainer(model = model)
    
    shap_values = explain(X = data)

    global _shap_val
    _shap_val = shap_values 

    vals = _shap_val.values

    if vals.ndim >= 3 and impact_type == 'all':
        vals = np.abs(vals).mean(axis = (0, 2))
    elif vals.ndim == 2 and impact_type == 'all':
        vals = np.abs(vals).mean(axis=0)

    elif (impact_type == "pos" or impact_type == "neg") and vals.ndim >=3:
        vals = vals.mean(axis=(0, 2))
    elif (impact_type == "pos" or impact_type == "neg") and vals.ndim == 2:
        vals = vals.mean(axis = 0)
    
    else:
        raise ValueError("Invalid Impact type or dimentions of shap values")

    for feature, value in zip(data.columns, vals):

            features.append(feature)
            all_values.append(value)

            if value < 0:
                neg_values.append(value)
                neg_feature.append(feature)
                
            else:
                pos_values.append(value)
                pos_feature.append(feature)
                
    if impact_type == 'neg':
            result = pd.DataFrame({
                        "Features" : neg_feature,
                        "Negative_Impact" : neg_values
                    }).sort_values(by="Negative_Impact", ascending=False)  
            
    elif impact_type == 'pos':
            result = pd.DataFrame({
                "Features" : pos_feature,
                "Positive_Impact" : pos_values

            }).sort_values(by="Positive_Impact", ascending=False)  
                
    elif impact_type == 'all':
            result = pd.DataFrame({
                "Features" : features,
                "Overall_Impact" : all_values
            }).sort_values(by="Overall_Impact", ascending=False)        
        
    else:
            raise ValueError("Invalid Impact type : must be neg, pos or all")
        
    return result, shap_values


def summary_plot(data):
    '''
    Summary plot for feature contribution for all the classes.
    '''
    
    shap.summary_plot(_shap_val, data)
    

def bar_plot(n_classes):
        '''
        Bar plot analysis of feature contribution for each class 
        '''
        for class_id in range(n_classes):    
            plt.title(f"For class_id {class_id}")
            shap.plots.bar(_shap_val[..., class_id])
            plt.tight_layout()
            plt.show()


# def force_plot(shap_values):
#       pass

    
def tree_plot(model , feature_names, size:tuple):
    plt.figure(figsize=size)
    tree.plot_tree(model, filled=True, feature_names=feature_names, class_names=True)
    plt.tight_layout()
    plt.show()









    
    



            

