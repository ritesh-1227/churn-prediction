import pandas as pd
from data_prep import DataPreprocessor
from feature_eng import FeatureEngineer, preprocess_and_engineer_features
from model import ChurnModel
from config import DATA_PATH
from sklearn.model_selection import train_test_split

def train_model():
    """
    Main training pipeline
    """
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Split data first
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize preprocessor and feature engineer
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    
    # Process training data
    X_train, y_train, preprocessor, feature_engineer = preprocess_and_engineer_features(
        train_df, 
        preprocessor, 
        feature_engineer, 
        is_training=True
    )
    
    # Initialize and train model
    churn_model = ChurnModel()
    churn_model.train(X_train, y_train, preprocessor, feature_engineer)
    # # Save all components
    # churn_model.save_model(preprocessor, feature_engineer)

    # Process test data using fitted transformers
    X_test, y_test = preprocess_and_engineer_features(
        test_df, 
        preprocessor, 
        feature_engineer, 
        is_training=False
    )
    
    # Evaluate model
    if y_test is not None:  # Add check for y_test
        print("Model Evaluation:")
        churn_model.evaluate(X_test, y_test)
    else:
        print("Warning: No target values available for evaluation")
    
    return preprocessor, feature_engineer, churn_model

if __name__ == "__main__":
    preprocessor, feature_engineer, churn_model = train_model()