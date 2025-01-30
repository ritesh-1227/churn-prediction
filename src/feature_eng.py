import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def create_features(self, df):
        """
        Create new features from existing ones
        """
        df_copy = df.copy()
        
        # Calculate average monthly charges
        df_copy['avg_monthly_charges'] = df_copy['total_charges'] / df_copy['tenure_months']
        
        # Create tenure-related features
        df_copy['tenure_year'] = df_copy['tenure_months'] / 12
        
        # Service count feature (number of services subscribed)
        service_columns = ['phone_service', 'multiple_lines', 'internet_service', 
                         'online_security', 'online_backup', 'device_protection',
                         'tech_support', 'streaming_tv', 'streaming_movies']
        
        df_copy['total_services'] = df_copy[service_columns].apply(
            lambda x: sum([1 for item in x if item not in ['No', 'No internet service', 0]])
            , axis=1
        )
        
        return df_copy

def preprocess_and_engineer_features(df, preprocessor, feature_engineer):
    """
    Combine preprocessing and feature engineering steps
    """
    # Preprocess data
    df_processed = preprocessor.preprocess(df)
    
    # Engineer features
    df_engineered = feature_engineer.create_features(df_processed)
    
    # Prepare final features
    X, y = preprocessor.prepare_features(df_engineered)
    
    return X, y