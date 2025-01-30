import os

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Telco_customer_churn.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'saved_models')

MODEL_FILENAME = 'churn_model.joblib'

# Create models directory if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Features
TARGET_COLUMN = 'churn_value'  # We'll use churn_value as our target
ID_COLUMN = 'customerid'

# Columns to exclude from modeling
EXCLUDE_COLUMNS = [
    'customerid', 
    'count', 
    'country', 
    'state', 
    'city', 
    'zip_code',
    'lat_long',
    'churn_label',
    'churn_reason',
    'churn_score',
    'cltv',
    'latitude',
    'longitude'
]