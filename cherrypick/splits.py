import pandas as pd
from sklearn.model_selection import train_test_split

def splitter(df, target:str, test_size:float) -> tuple[tuple, tuple]:
        """
        Split dataset into training and testing sets.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset containing features and target variable.

        target : str
            Column name of the target variable.

        test_size : float
            Proportion of the dataset to include in the test split.
            Must be between 0.0 and 1.0.

        Returns
        -------
        tuple
            A tuple containing:

            - X_train - pandas.DataFrame
            - X_test  - pandas.DataFrame
            - y_train - pandas.Series
            - y_test  - pandas.Series

        Notes
        -----
        - Features are obtained by dropping the target column from ``df``.
        - Internally uses ``train_test_split`` from scikit-learn.
        """
        
        try:
            X = df.drop(columns=target)
            y = df[target]
        
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            train_data = (X_train, y_train)
            test_data = (X_test, y_test)

            print("_____________Shape - Train vs Test_____________")
            print(f"Train dataset :\n\nDependent feature = {X_train.shape}\nIndependent Feature = {y_train.shape}")
            print(f"Test dataset :\n\nDependent feature = {X_test.shape}\nIndependent Feature = {y_test.shape}")

            return train_data, test_data

        except Exception as err:
            raise Exception(f"{err}")
        


   