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
        """
        df_copy = df.copy()
        
        # Remove unwanted columns
        cols_to_drop = EXCLUDE_COLUMNS
        existing_cols_to_drop = [col for col in cols_to_drop if col in df_copy.columns]
        if existing_cols_to_drop:
            df_copy = df_copy.drop(existing_cols_to_drop, axis=1)
        
        # Handle target column
        if 'churn_value' in df_copy.columns:
            y = df_copy['churn_value']
            X = df_copy.drop('churn_value', axis=1)
        else:
            y = None
            X = df_copy
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y