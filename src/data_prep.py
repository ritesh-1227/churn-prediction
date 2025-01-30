import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from config import EXCLUDE_COLUMNS

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        
    def preprocess(self, df, is_training=True):
        """
        Preprocess the dataframe for modeling
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        is_training (bool): Whether this is training data or prediction data
        """
        df_copy = df.copy()
        
        # Convert column names to snake case
        df_copy.columns = [col.lower().replace(' ', '_') for col in df_copy.columns]
        
        # Convert total_charges to numeric
        df_copy['total_charges'] = pd.to_numeric(df_copy['total_charges'], errors='coerce')
        
        # Handle missing values in total_charges
        df_copy['total_charges'].fillna(df_copy['monthly_charges'] * df_copy['tenure_months'], 
                                      inplace=True)
        
        # Handle lat_long column (split if needed)
        if 'lat_long' in df_copy.columns:
            df_copy.drop('lat_long', axis=1, inplace=True)
            
        # Convert categorical variables
        categorical_columns = df_copy.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['customerid', 'churn_reason']:  # Skip ID and free text columns
                if is_training or not self.is_fitted:
                    self.label_encoders[col] = LabelEncoder()
                    df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col])
                else:
                    if col in self.label_encoders:
                        try:
                            df_copy[col] = self.label_encoders[col].transform(df_copy[col])
                        except ValueError:
                            # Handle unseen categories
                            df_copy[col] = self.label_encoders[col].transform([self.label_encoders[col].classes_[0]] * len(df_copy))
        
        if is_training:
            self.is_fitted = True
            
        return df_copy
    
    def prepare_features(self, df, is_training=True):
        """
        Prepare features for modeling
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        is_training (bool): Whether this is training data or prediction data
        """
        df_copy = df.copy()
        
        # Remove columns if they exist
        cols_to_drop = EXCLUDE_COLUMNS

        # ['customerid', 'churn_label', 'churn_reason', 
        #                'count', 'country', 'state', 'city', 'zip_code']
        
        # Only drop columns that exist
        existing_cols_to_drop = [col for col in cols_to_drop if col in df_copy.columns]
        if existing_cols_to_drop:
            df_copy = df_copy.drop(existing_cols_to_drop, axis=1)
        
        if is_training:
            # Training data case - remove target columns and get target
            target_cols = ['churn_value', 'churn_score', 'cltv']
            existing_target_cols = [col for col in target_cols if col in df_copy.columns]
            
            if existing_target_cols:
                X = df_copy.drop(existing_target_cols, axis=1)
                y = df_copy['churn_value']
            else:
                raise ValueError("Target column 'churn_value' not found in training data")
                
            # Fit and transform features
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled, y
        else:
            # Prediction data case - just transform features
            if any(col in df_copy.columns for col in ['churn_value', 'churn_score', 'cltv']):
                df_copy = df_copy.drop([col for col in ['churn_value', 'churn_score', 'cltv'] 
                                      if col in df_copy.columns], axis=1)
            
            X_scaled = self.scaler.transform(df_copy)
            return X_scaled, None

def load_and_split_data(filepath, test_size=0.2, random_state=42):
    """
    Load data and split into train and test sets
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess data
    df_processed = preprocessor.preprocess(df, is_training=True)
    
    # Prepare features
    X, y = preprocessor.prepare_features(df_processed, is_training=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, preprocessor