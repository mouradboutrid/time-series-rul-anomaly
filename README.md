# Time Series – RUL Forecasting and Anomaly Detection on CMAPSS Dataset

This project implements a comprehensive predictive maintenance system using the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. It focuses on three critical predictive maintenance tasks:

- **XGBoost regression** for Remaining Useful Life (RUL) prediction
- **LSTM autoencoder** for anomaly detection
- **LSTM-based sequence model** for RUL forecasting

The project leverages advanced time series modeling, machine learning, and deep learning techniques to enhance maintenance decision-making for turbofan engines, potentially reducing downtime and maintenance costs.

## Project Structure

```
time-series-rul-anomaly/
│
├── data/                               # Raw and preprocessed CMAPSS data
│   ├── raw/                            # Original NASA CMAPSS dataset files
│   └── processed/                      # Preprocessed and feature-engineered data
│
├── models/                             # Saved model files
│   ├── xgboost/                        # XGBoost model artifacts
│   ├── lstm_autoencoder/               # LSTM autoencoder model artifacts
│   └── lstm_forecaster/                # LSTM forecaster model artifacts
│
├── notebooks/
│   ├── 01_data_preparation.ipynb       # Data loading and preprocessing
│   ├── 02_xgboost_rul_prediction.ipynb # XGBoost model training and evaluation
│   ├── 03_lstm_autoencoder_anomaly_detection.ipynb
│   ├── 04_lstm_forecaster_rul_prediction.ipynb
│   └── 05_model_evaluation.ipynb       # Comprehensive evaluation and visualization
│
├── scripts/
│   ├── data_utils.py                   # Data loading and preprocessing utilities
│   ├── feature_engineering.py          # Feature extraction and engineering functions
│   ├── model_utils.py                  # Model training and evaluation utilities
│   └── visualization.py                # Plotting and visualization functions
│
├── results/                            # Performance metrics and visualizations
│   ├── figures/                        # Generated plots and charts
│   └── metrics/                        # Model performance metrics
│
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package installation script
├── .gitignore                          # Git ignore file
└── README.md                           # This file
```

## Models Overview

### 1. XGBoost RUL Prediction

- **Purpose**: Regression model to predict the Remaining Useful Life (RUL) of engines from time series sensor data
- **Features**:
  - Engineered features from raw sensor readings
  - Statistical features (mean, std, min, max) within sliding windows
  - Trend indicators and rate of change features
- **Approach**:
  - Feature importance analysis to select most predictive sensors
  - Hyperparameter optimization using grid search with cross-validation
  - RUL score optimization using custom metrics

### 2. LSTM Autoencoder for Anomaly Detection

- **Purpose**: Detect abnormal behavior patterns that may indicate impending failures
- **Architecture**:
  - Encoder: Multiple LSTM layers with decreasing dimensionality
  - Bottleneck layer: Dense representation of normal behavior
  - Decoder: Multiple LSTM layers reconstructing the input sequence
- **Detection Method**:
  - Training on normal operation data only
  - Anomaly scoring based on reconstruction error
  - Dynamic thresholding to adapt to different operational conditions

### 3. LSTM RUL Forecasting Model

- **Purpose**: Forecast RUL as a future sequence rather than a single point estimate
- **Architecture**:
  - Sequence-to-sequence LSTM neural network
  - Bidirectional LSTM layers for capturing temporal patterns
  - Dense output layers with ReLU activation for non-negative RUL values
- **Training**:
  - Teacher forcing during training phase
  - Custom loss function weighing recent time steps more heavily
  - Learning rate scheduling for convergence

## Dataset

- **Source**: [NASA Prognostics Data Repository - CMAPSS Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
- **Description**: Simulated data from turbofan engine degradation simulations
- **Subset used**: FD004 (multiple operational conditions and fault modes)
- **Contents**:
  - Training data: Full run-to-failure trajectories
  - Test data: Truncated engine trajectories for RUL prediction
  - 21 sensor measurements per time step
  - 3 operational setting conditions
  - Multiple fault modes

## Usage

### Clone the Repository

```bash
git clone https://github.com/mouradboutrid/time-series-rul-anomaly.git
cd time-series-rul-anomaly
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data Preparation

1. Download the CMAPSS dataset from the [NASA repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
2. Place the raw data files in the `data/raw/` directory
3. Run the data preparation notebook:

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

### Run Notebooks in Order

Open and run the notebooks in the following order:

1. `01_data_preparation.ipynb` - Data loading, preprocessing, and feature engineering
2. `02_xgboost_rul_prediction.ipynb` - Train and evaluate the XGBoost model
3. `03_lstm_autoencoder_anomaly_detection.ipynb` - Train and evaluate the LSTM autoencoder
4. `04_lstm_forecaster_rul_prediction.ipynb` - Train and evaluate the LSTM forecaster
5. `05_model_evaluation.ipynb` - Comprehensive evaluation and comparison of models

### Using the Models for Prediction

```python
# Example code to load and use the trained XGBoost model
import joblib
from scripts.data_utils import preprocess_data

# Load the model
model = joblib.load('models/xgboost/xgb_rul_predictor.pkl')

# Preprocess new data
X_new = preprocess_data(new_data)

# Make predictions
predictions = model.predict(X_new)
```

## Implementation Details

### Feature Engineering

The project implements several feature engineering techniques:
- **Window-based features**: Statistical measures over sliding windows
- **Trend indicators**: Slope and curvature of sensor readings
- **Interaction features**: Products of correlated sensors
- **Normalization**: Min-max scaling based on operational conditions

### Hyperparameter Optimization

- **XGBoost**: Grid search with cross-validation
- **LSTM Models**: Bayesian optimization with early stopping

### Evaluation Metrics

- **Regression Tasks**:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Scoring function specific to RUL prediction
- **Anomaly Detection**:
  - Precision, Recall, F1-Score
  - Area Under ROC Curve (AUC)
  - Time to Detection (TTD)

## Results

Detailed evaluation metrics and example plots can be found in `05_model_evaluation.ipynb`. Key findings include:

- XGBoost model achieves RMSE of approximately 15-20 cycles on the test set
- LSTM autoencoder can detect anomalies approximately 30-50 cycles before failure
- LSTM forecaster provides accurate RUL trajectories up to 50 cycles into the future

## Future Work

- **Model Improvements**:
  - Add attention mechanism to LSTM forecaster
  - Implement transformer-based architecture for sequence modeling
  - Integrate quantile regression for uncertainty estimation
  
- **System Enhancements**:
  - Real-time prediction pipeline
  - Interactive visualization dashboard
  - Integration with maintenance scheduling systems
  
- **Methodological Extensions**:
  - Transfer learning between different engine types
  - Explainable AI techniques for maintenance decision support
  - Reinforcement learning for optimal maintenance scheduling

## Dependencies

- Python 3.8+
- Data Processing:
  - NumPy (1.20+)
  - Pandas (1.3+)
  - SciPy (1.7+)
- Machine Learning:
  - Scikit-learn (1.0+)
  - XGBoost (1.5+)
  - TensorFlow (2.6+) / Keras
- Visualization:
  - Matplotlib (3.4+)
  - Seaborn (0.11+)
  - Plotly (5.3+)

All dependencies are listed in `requirements.txt`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- NASA Prognostics Center for the CMAPSS dataset
- Open-source libraries: XGBoost, TensorFlow, Keras, Scikit-learn
