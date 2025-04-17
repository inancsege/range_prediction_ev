import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest


def preprocess_data(df, key_columns, features):
    """Clean and preprocess the data."""
    # Convert key columns to numeric and drop rows with missing values in these columns
    df[key_columns] = df[key_columns].apply(pd.to_numeric, errors='coerce')
    df_clean = df.dropna(subset=key_columns)
    
    # Define preprocessing for numerical features
    numerical_features = df_clean.select_dtypes(include=['int64', 'float64']).columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_features = df_clean.select_dtypes(include=['object']).columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Apply transformations
    X = preprocessor.fit_transform(df_clean)
    y = df_clean['trip_distance(km)']
    
    # Feature selection: Remove features with low variance
    selector = VarianceThreshold(threshold=0.1)
    X = selector.fit_transform(X)
    
    # Outlier detection and removal
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X)
    mask = yhat != -1
    X, y = X[mask, :], y[mask]
    
    return X, y, df_clean


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