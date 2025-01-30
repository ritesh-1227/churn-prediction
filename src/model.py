import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from config import MODEL_FILENAME, MODEL_PATH

class ChurnModel:
    def __init__(self, model_path=MODEL_PATH):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model_path = model_path
        
    def train(self, X_train, y_train, preprocessor):
        """
        Train the model and save preprocessor
        """
        self.model.fit(X_train, y_train)
        # Save preprocessor along with model
        self.save_model(preprocessor)
        
    def predict(self, X):
        """
        Make predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        predictions = self.predict(X_test)
        prob_predictions = self.predict_proba(X_test)[:, 1]
        
        print("Classification Report:")
        print(classification_report(y_test, predictions))
        
        print("\nROC-AUC Score:", roc_auc_score(y_test, prob_predictions))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': range(X_test.shape[1]),
            'importance': self.model.feature_importances_
        })
        print("\nTop 10 Important Features:")
        print(feature_importance.sort_values('importance', ascending=False).head(10))
        
    def save_model(self, preprocessor, filename=MODEL_FILENAME):
        """
        Save model and preprocessor to disk
        """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        # Save both model and preprocessor
        artifacts = {
            'model': self.model,
            'preprocessor': preprocessor
        }
        joblib.dump(artifacts, os.path.join(self.model_path, filename))
        
    def load_model(self, filename=MODEL_FILENAME):
        """
        Load model and preprocessor from disk
        """
        artifacts = joblib.load(os.path.join(self.model_path, filename))
        if not isinstance(artifacts, dict):
            raise TypeError(f"Unexpected data type {type(artifacts)} in joblib file.")
    
        preprocessor = artifacts.get('preprocessor')
        model = artifacts.get('model')
        
        if preprocessor is None or model is None:
            raise ValueError("Missing 'preprocessor' or 'model' in loaded artifacts.")
    
        return preprocessor, model