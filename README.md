# Time Series – RUL Forecasting and Anomaly Detection on CMAPSS Dataset

This project implements a predictive maintenance system using the CMAPSS dataset. It includes three main models:

- **XGBoost regression** for Remaining Useful Life (RUL) prediction
- **LSTM autoencoder** for anomaly detection
- **LSTM-based sequence model** for RUL forecasting

The project leverages time series modeling and deep learning techniques to enhance maintenance decisions for turbofan engines.

---

## Project Structure
time-series-rul-anomaly/
│
├── data/ # Raw and preprocessed CMAPSS data
├── models/ # Saved model files (optional)
├── notebooks/
│ ├── 01_data_preparation.ipynb # Data loading and preprocessing
│ ├── 02_xgboost_rul_prediction.ipynb
│ ├── 03_lstm_autoencoder_anomaly_detection.ipynb
│ ├── 04_lstm_forecaster_rul_prediction.ipynb
│ ├── 05_model_evaluation.ipynb # Evaluation and visualization
├── scripts/
│ └── data_utils.py # Reusable functions (e.g., load_data)
├── requirements.txt # Python dependencies
├── README.md
├── .gitignore


---

## Models Overview

### 1. XGBoost RUL Prediction
- Trains an XGBoost regressor on engineered features.
- Predicts the Remaining Useful Life (RUL) of engines from time series data.

### 2. LSTM Autoencoder for Anomaly Detection
- Learns normal behavior via sequence reconstruction.
- Detects anomalies based on reconstruction error.

### 3. LSTM RUL Forecasting Model
- Forecasts RUL as a future sequence.
- Uses a sequence-to-sequence LSTM neural network.

---

## Usage

### Clone the Repository

```bash
git clone https://github.com/mouradboutrid/time-series-rul-anomaly.git
cd time-series-rul-anomaly
pip install -r requirements.txt

Dataset
Source: CMAPSS – NASA Prognostics Data Repository

Subset used: FD004 (multiple operational conditions and fault modes)

Dependencies
Python 3.x

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

XGBoost

TensorFlow / Keras

All dependencies are listed in requirements.txt.

Results
Evaluation metrics and example plots can be found in 05_model_evaluation.ipynb. These include:

RUL prediction error (e.g., RMSE, MAE)

Forecast quality over time

Anomaly reconstruction error plots

Future Work
Add attention mechanism to LSTM forecaster

Integrate quantile regression for uncertainty estimation

Deploy models for real-time inference

License
This project is licensed under the MIT License.


