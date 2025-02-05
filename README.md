# Customer Churn Prediction System

## Project Overview
This project implements a machine learning system to predict customer churn probability. Using various customer attributes and behavior patterns, the model assesses the likelihood of a customer leaving the service.

## Features
- **Data Preprocessing Pipeline**: Handles missing values, encoding, and feature scaling
- **Feature Engineering**: Creates derived features to improve model performance
- **Machine Learning Model**: Random Forest Classifier for churn prediction
- **Web Interface**: Interactive Streamlit app for real-time predictions
- **Modular Design**: Separate components for preprocessing, feature engineering, and modeling

## Project Structure
```plaintext
churn_prediction/
├── data/
│   └── Telco_customer_churn.csv
├── models/
│   └── saved_models/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration parameters
│   ├── data_prep.py       # Data preprocessing
│   ├── feature_eng.py     # Feature engineering
│   ├── model.py           # Model training
│   └── utils.py           # Helper functions
├── app/
│   └── streamlit_app.py   # Streamlit interface
├── requirements.txt
└── README.md
```


## Installation & Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd churn_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python src/train.py
```

4. Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

## Usage

### Web Interface
The Streamlit interface allows users to:
- Input customer information
- Get real-time churn predictions
- View prediction confidence levels
- Access basic model insights

### Model Features
The system considers various customer attributes:
- Demographics (age, gender)
- Service usage patterns
- Contract information
- Billing details
- Service subscriptions

## Deployment
The application is deployed on Streamlit Cloud and can be accessed at: [your-app-url]

## Future Improvements
- [ ] Batch prediction capabilities
- [ ] More detailed feature importance analysis
- [ ] Additional visualization components
- [ ] Model performance monitoring
- [ ] Customer retention recommendations

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset source: IBM Telco Customer Churn dataset
- Streamlit for the web framework
- Various open-source packages that made this possible
