import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest


class Preprocessor:
    def __init__(self, df, key_columns, features):
        self.df = df
        self.key_columns = key_columns
        self.features = features
        self.df_clean = None
        self.X = None
        self.y = None

    def clean_data(self):
        """Convert key columns to numeric and drop rows with missing values in these columns."""
        self.df[self.key_columns] = self.df[self.key_columns].apply(pd.to_numeric, errors='coerce')
        self.df_clean = self.df.dropna(subset=self.key_columns)

    def preprocess_numerical_features(self):
        """Preprocess numerical features."""
        numerical_features = self.df_clean.select_dtypes(include=['int64', 'float64']).columns
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        return numerical_transformer, numerical_features

    def preprocess_categorical_features(self):
        """Preprocess categorical features."""
        categorical_features = self.df_clean.select_dtypes(include=['object']).columns
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        return categorical_transformer, categorical_features

    def apply_transformations(self):
        """Apply transformations to the data."""
        num_transformer, num_features = self.preprocess_numerical_features()
        cat_transformer, cat_features = self.preprocess_categorical_features()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)
            ]
        )
        self.X = preprocessor.fit_transform(self.df_clean)
        self.y = self.df_clean['trip_distance(km)']

    def feature_selection(self):
        """Remove features with low variance."""
        selector = VarianceThreshold(threshold=0.1)
        self.X = selector.fit_transform(self.X)

    def remove_outliers(self):
        """Detect and remove outliers."""
        iso = IsolationForest(contamination=0.1)
        yhat = iso.fit_predict(self.X)
        mask = yhat != -1
        self.X, self.y = self.X[mask, :], self.y[mask]

    def preprocess(self):
        """Run all preprocessing steps."""
        self.clean_data()
        self.apply_transformations()
        self.feature_selection()
        self.remove_outliers()
        return self.X, self.y, self.df_clean


def preprocess_data(df, key_columns, features):
    """Clean and preprocess the data."""
    preprocessor = Preprocessor(df, key_columns, features)
    return preprocessor.preprocess()


def prepare_data(data, target_column, drop_columns):
    """Prepare data by dropping specified columns and encoding categorical variables."""
    # Drop specified columns
    X = data.drop(drop_columns, axis=1)
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=categorical_cols)
    
    # Extract target variable
    Y = data[target_column]
    
    return X, Y 