# 🏎 Formula 1 Lap Time Prediction System

An end-to-end Machine Learning web application that predicts Formula 1 lap times using historical Bahrain Grand Prix data (2021–2024).

This project integrates data collection, feature engineering, hybrid ensemble modeling, and web deployment using Flask.

---

## 📌 Project Overview

This system predicts lap time (in seconds) based on race-related inputs such as:

- Year
- Lap Number
- Stint
- Tyre Life
- Fresh Tyre Indicator
- Race Position
- Tyre Compound
- Team
- Driver

The trained model is integrated into a Flask web application that allows real-time lap time prediction through a user-friendly interface.

---

## 🧠 Machine Learning Approach

### Data Source
- FastF1 API  
- Bahrain Grand Prix race data (2021–2024 seasons)

### Feature Engineering
The following features were used for training:

- Lap Number
- Stint
- Tyre Life
- Tyre Compound
- Driver
- Team
- Race Position
- Season Year

Categorical variables were encoded using one-hot encoding.

---

### Model Architecture

Hybrid Ensemble Regression Model:
- Random Forest Regressor
- XGBoost Regressor
- Combined using Voting Regressor



### Evaluation Metrics

The model performance was evaluated using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score



## 🌐 Web Application

Built using:

- Flask (Backend Framework)
- HTML & CSS (Frontend UI)
- Joblib (Model Serialization)

The web interface allows users to input race parameters and receive predicted lap times instantly.

---

## 📂 Project Structure

project/
│
├── app.py
├── bahrain_hybrid_model.pkl
├── bahrain_model_columns.pkl
├── requirements.txt
│
├── templates/
│     └── index.html
│
└── README.md


## 🚀 Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- FastF1 API



## 📈 Future Improvements

- Extend prediction to multiple circuits
- Add feature importance visualization in the UI
- Implement cross-validation tuning
- Deploy to cloud platform

