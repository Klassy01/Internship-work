import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_outliers=["Family member count", "Income", "Employment length"]):
        self.feat_with_outliers = feat_with_outliers
        self.Q1 = None
        self.Q3 = None
        self.IQR = None

    def fit(self, X, y=None):
        if set(self.feat_with_outliers).issubset(X.columns):
            self.Q1 = X[self.feat_with_outliers].quantile(0.25)
            self.Q3 = X[self.feat_with_outliers].quantile(0.75)
            self.IQR = self.Q3 - self.Q1
        return self

    def transform(self, X):
        X = X.copy()
        if hasattr(self, 'Q1') and self.Q1 is not None:
            condition = ~((X[self.feat_with_outliers] < (self.Q1 - 3 * self.IQR)) | 
                         (X[self.feat_with_outliers] > (self.Q3 + 3 * self.IQR))).any(axis=1)
            X = X[condition]
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Ensure only the expected features are present"""
    def __init__(self, expected_features):
        self.expected_features = expected_features
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Add missing features as zeros
        for feature in self.expected_features:
            if feature not in X.columns:
                X[feature] = 0
        # Remove unexpected features
        return X[self.expected_features]

def get_feature_names(column_transformer):
    """Get feature names from a ColumnTransformer"""
    output_features = []
    
    for name, transformer, features in column_transformer.transformers_:
        if transformer == 'drop':
            continue
        if hasattr(transformer, 'get_feature_names'):
            names = transformer.get_feature_names(features)
        elif isinstance(transformer, _VectorizerMixin):
            names = transformer.get_feature_names()
        elif isinstance(transformer, SelectorMixin):
            names = [features[i] for i in transformer.get_support(indices=True)]
        else:
            names = features
        output_features.extend(names)
    
    return output_features

def full_pipeline(df, training=True):
    """Complete preprocessing pipeline that maintains feature consistency"""
    df = df.copy()
    
    # Store original columns for reference
    original_columns = df.columns.tolist()
    
    # Separate target if it exists
    y = None
    if "Is high risk" in df.columns:
        y = df["Is high risk"]
        X = df.drop(columns=["Is high risk"])
    else:
        X = df
    
    # Define columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Build transformers
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    
    # Create full pipeline
    pipeline = Pipeline([
        ("outlier_removal", OutlierRemover()),
        ("preprocessing", preprocessor)
    ])
    
    # Fit or transform based on mode
    if training:
        X_processed = pipeline.fit_transform(X)
        feature_names = get_feature_names(preprocessor)
    else:
        X_processed = pipeline.transform(X)
        feature_names = get_feature_names(preprocessor)
    
    # Create DataFrame with proper column names
    processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # Add target back if it exists
    if y is not None:
        processed_df["Is high risk"] = y.reset_index(drop=True)
    
    return processed_df, pipeline