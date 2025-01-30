import pandas as pd
from data_prep import DataPreprocessor, load_and_split_data
from feature_eng import FeatureEngineer, preprocess_and_engineer_features
from model import ChurnModel
from config import DATA_PATH

def train_model():
    """
    Main training pipeline
    """
    # Load and split data
    X_train, X_test, y_train, y_test, preprocessor = load_and_split_data(DATA_PATH)
    
    # Initialize and train model
    churn_model = ChurnModel()
    churn_model.train(X_train, y_train, preprocessor)  # This will save both model and preprocessor
    
    # Evaluate model
    print("Model Evaluation:")
    churn_model.evaluate(X_test, y_test)
    
    return preprocessor, churn_model

if __name__ == "__main__":
    preprocessor, model = train_model()