import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_prep import DataPreprocessor

class FeatureEngineer:
    def __init__(self):
        self.feature_stats = {}  # Store statistics from training data
    
    def fit(self, df):
        """Learn parameters from training data"""
        # Store any statistics needed for feature creation
        self.feature_stats['avg_monthly_charges'] = df['monthly_charges'].mean()
        self.feature_stats['avg_tenure'] = df['tenure_months'].mean()
        return self
        
    def transform(self, df):
        """Apply feature engineering using learned parameters"""
        df_copy = df.copy()
        
        # Calculate average charges per month
        df_copy['avg_monthly_charges'] = df_copy['total_charges'] / df_copy['tenure_months'].replace(0, 1)
        
        # Create tenure categories
        df_copy['tenure_year'] = df_copy['tenure_months'] / 12
        
        # Service count feature
        service_columns = ['phone_service', 'multiple_lines', 'internet_service', 
                         'online_security', 'online_backup', 'device_protection',
                         'tech_support', 'streaming_tv', 'streaming_movies']
        
        df_copy['total_services'] = df_copy[service_columns].apply(
            lambda x: sum([1 for item in x if item not in ['No', 'No internet service', 0]]), axis=1
        )
        
        return df_copy
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)

def preprocess_and_engineer_features(df, preprocessor, feature_engineer, is_training=True):
    """
    Combine preprocessing and feature engineering steps
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data
    preprocessor : DataPreprocessor
        Preprocessor instance
    feature_engineer : FeatureEngineer
        Feature engineer instance
    is_training : bool
        Whether this is training data or test/prediction data
    """
    # Preprocess data
    if is_training:
        df_processed = preprocessor.preprocess(df, is_training=True)
        df_engineered = feature_engineer.fit_transform(df_processed)
        X, y = preprocessor.prepare_features(df_engineered, is_training=True)
        return X, y, preprocessor, feature_engineer
    else:
        df_processed = preprocessor.preprocess(df, is_training=False)
        df_engineered = feature_engineer.transform(df_processed)
        X, y = preprocessor.prepare_features(df_engineered, is_training=False)
        return X, y