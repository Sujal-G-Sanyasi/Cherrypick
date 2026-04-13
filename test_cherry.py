import pandas as pd
import numpy as np
from cherrypick.explain import explainer
from cherrypick import Orchestrator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from cherrypick.splits import splitter
from cherrypick.anomaly import OutlierPruner
from cherrypick.preprocessing import Preprocessor

import warnings
warnings.filterwarnings('ignore')


def test_explainer():

    df = pd.DataFrame({
    "feature_1": [12, 45, 23, 67, 34, 89, 54, 21, 76, 38, 49, 62, 28, 90, 55],
    "feature_2": [100, 200, 150, 300, 250, 400, 350, 120, 370, 220, 260, 310, 180, 420, 360],
    "feature_3": [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    "feature_4": [5, 3, 6, 2, 7, 1, 8, 4, 9, 5, 6, 7, 3, 2, 8],
    "feature_5": [10, 20, 15, 25, 18, 30, 28, 12, 27, 19, 22, 24, 12, 35, 29],
    # "target":     [100, 111, 222, 500, 899, 641, 112, 400, 10000, 80000, 190, 133, 112, 41, 1]
    "target" : [1, 1, 1, 2, 1, 1, 1, 0, 2, 1, 2, 1, 1, 2, 0]
})
    X= df.drop(columns='target')
    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    
    result, shap_vals = explainer(model=rf, impact_type="pos", data=X_test)

    assert not result.empty
    assert shap_vals.values.size > 0
    assert shap_vals.values.ndim >= 2


def test_orchestration_regression():
   
    X, y = make_regression(
    n_samples=100,
    n_features=5,
    n_informative=3,
    random_state=42
)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    cherry = Orchestrator(
        problem_statement='regression',
        focus_regressor='rmse',
        train=(X_train, y_train),
        test=(X_test, y_test),
        file_dir='model'
    )
    model = cherry.orchestrate()
    
    assert model is not None and isinstance(model, type(cherry.topkmodel(access_estimator=1))) 
    assert isinstance(model, type(cherry.best_estimator))


def test_orchestration_classify():
   
    X, y = make_classification(
    n_samples=100,
    n_features=5,
    n_classes=3,
    n_informative=3,
    random_state=42
)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    cherry = Orchestrator(
        problem_statement='classification',
        focus_classifier='precision',
        train=(X_train, y_train),
        test=(X_test, y_test),
        file_dir='model'
    )
    model = cherry.orchestrate()
    
    assert isinstance(model, type(cherry.topkmodel(access_estimator=1)))
    assert isinstance(model, type(cherry.best_estimator))

def test_splitter():
    df = pd.DataFrame({
    "feature_1": [12, 45, 23, 67, 34, 89, 54, 21, 76, 38, 49, 62, 28, 90, 55],
    "feature_2": [100, 200, 150, 300, 250, 400, 350, 120, 370, 220, 260, 310, 180, 420, 360],
    "feature_3": [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    "feature_4": [5, 3, 6, 2, 7, 1, 8, 4, 9, 5, 6, 7, 3, 2, 8],
    "feature_5": [10, 20, 15, 25, 18, 30, 28, 12, 27, 19, 22, 24, 12, 35, 29],
    "target" : [1, 2, 0, 0, 1, 2, 1, 0, 2, 1, 2, 1, 1, 2, 0]
})
    train, test = splitter(df=df, target='target', test_size=0.2)

    assert type(train) == tuple and not None
    assert type(test) == tuple and not None

def test_anomaly():
    df =  pd.DataFrame({
    "feature_1": [12, 45, 23, 67, 34, 89, 54, 21, 76, 38, 49, 62, 28, 90, 55],
    "feature_2": [100, 200, 150, 300, 250, 400, 350, 120, 370, 220, 260, 310, 180, 4200000, 3600000],
    "feature_3": [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    "feature_4": [5, 3, 6, 2, 7, 1, 8, 4, 9, 5, 6, 7, 3, 2, 8],
    "feature_5": [10, 20, 15, 25, 18, 30, 28, 12, 27, 19, 22, 24, 12, 35, 29],
    "target" : [1, 2, 0, 0, 1, 2, 1, 0, 2, 1, 2, 1, 1, 2, 0]
})

    prune = OutlierPruner(
        df=df,
        method = 'isoforest',
        col = 'feature_2'
    )

    df_pruned = prune.remove_outlier()

    pruned_data = df_pruned[df_pruned['feature_2']>1000]

    assert df_pruned.shape <= df.shape and not df_pruned.empty
    assert pruned_data.empty or df_pruned.shape <= df.shape


def test_oneHotEncoder():
    df= pd.DataFrame({
    "feature_1": [12, 45, 23, 67, 34, 89, 54, 21, 76, 38, 49, 62, 28, 90, 55],
    "feature_2": [100, 200, 150, 300, 250, 400, 350, 120, 370, 220, 260, 310, 180, 4200000, 3600000],
    "feature_3": ['Bad', 'good', 'bad', 'good', 'bad', 'good', 'good', 'bad', 'good', 'bad', 'good', 'bad', 'good', 'bad', 'good'],
    "feature_4": [5, 3, 6, 2, 7, 1, 8, 4, 9, 5, 6, 7, 3, 2, 8],
    "feature_5": [10, 20, 15, 25, 18, 30, 28, 12, 27, 19, 22, 24, 12, 35, 29],
    "target" : [1, 2, 0, 0, 1, 2, 1, 0, 2, 1, 2, 1, 1, 2, 0]
})
    preprocess = Preprocessor(
        df=df
    )
    train, test = splitter(df=df, test_size=0.2, target='target')
    X_train, X_test = preprocess.encoder(train_data=train, test_data=test, column='feature_3', encoder_dir='model', type='onehot')

    train = X_train.select_dtypes(exclude='object')['feature_3_bad'].dtypes
    test = X_test.select_dtypes(exclude='object')['feature_3_bad'].dtypes
 
    assert train != 'object' and test != 'object'

def test_labelEncoder():
    
    df= pd.DataFrame({
    "feature_1": [12, 45, 23, 67, 34, 89, 54, 21, 76, 38, 49, 62, 28, 90, 55],
    "feature_2": [100, 200, 150, 300, 250, 400, 350, 120, 370, 220, 260, 310, 180, 4200000, 3600000],
    "feature_3": ['Bad', 'good', 'bad', 'good', 'bad', 'good', 'good', 'bad', 'good', 'bad', 'good', 'bad', 'good', 'bad', 'good'],
    "feature_4": [5, 3, 6, 2, 7, 1, 8, 4, 9, 5, 6, 7, 3, 2, 8],
    "feature_5": [10, 20, 15, 25, 18, 30, 28, 12, 27, 19, 22, 24, 12, 35, 29],
    "target" : ['b', 'a', 'c', 'b', 'a', 'c', 'a', 'b', 'c', 'a', 'c', 'b', 'a', 'a', 'b']
})
    preprocess = Preprocessor(
        df=df
    )
    train, test = splitter(df=df, test_size=0.2, target='target')
    y_train, y_test = preprocess.encoder(type='label', train_data=train, test_data=test, column='feature_3', encoder_dir='model')

    assert np.issubdtype(y_train.dtype, np.integer) and np.issubdtype(y_test.dtype, np.integer)


def test_imputer():
    df= pd.DataFrame({
    "feature_1": [12, 45, 23, 67, 34, 89, 54, 21, 76, 38, 49, 62, 28, 90, 55],
    "feature_2": [100, 200, 150, None, 250, 400, 350, 120, 370, 220, 260, 310, 180, 4200000, 3600000],
    "feature_3": ['Bad', 'good', 'bad', 'good', 'bad', 'good', 'good', 'bad', 'good', 'bad', 'good', 'bad', 'good', 'bad', 'good'],
    "feature_4": [5, 3, 6, 2, 7, 1, 8, 4, 9, None, 6, 7, 3, 2, 8],
    "feature_5": [10, 20, 15, 25, 18, 30, 28, 12, 27, None, 22, 24, 12, 35, 29],
    "target" : ['b', 'a', 'c', 'b', 'a', 'c', 'a', 'b', 'c', 'a', 'c', 'b', 'a', 'a', 'b']
})
    preprocessor = Preprocessor(
        df = df
    )

    preprocessor = Preprocessor(df=df)

    df_processed = preprocessor.fill_null(type='mean', columns='feature_2')
    df_processed = preprocessor.fill_null(type='median', columns='feature_5')
    df_processed = preprocessor.fill_null(type='mode', columns='feature_4')

    assert df_processed[['feature_2', 'feature_4', 'feature_5']].isna().sum().sum() == 0

    