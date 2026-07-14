# Flight Price Prediction

A machine learning project that predicts flight ticket prices in India based on features like airline, source, destination, number of stops, duration, and travel date.

## Overview

This project walks through a full ML pipeline:
- Exploratory Data Analysis (EDA)
- Data cleaning and feature engineering
- Categorical encoding (ordinal mapping for stops, one-hot encoding for airline/source/destination/additional info)
- Model training with a Random Forest Regressor
- Log-transforming the target variable to improve performance
- Model evaluation and persistence

## Dataset

The dataset contains historical flight booking data with the following raw features:
- `Airline`, `Source`, `Destination`, `Route`
- `Date_of_Journey`, `Dep_Time`, `Arrival_Time`, `Duration`
- `Total_Stops`, `Additional_Info`
- `Price` (target variable)

## Feature Engineering

- Extracted `Year`, `Month`, `Day` from `Date_of_Journey`
- Extracted `Arrival_hour`, `Arrival_min`, `Departure_hour`, `Departure_min` from time columns
- Converted `Duration` (e.g. `"2h 50m"`) into total minutes
- Mapped `Total_Stops` to ordinal integers (`non-stop` → 0, `1 stop` → 1, etc.) since stop count has a natural order
- One-hot encoded `Airline`, `Source`, `Destination`, and `Additional_Info`, since these are unordered (nominal) categories

## Model

- **Algorithm:** Random Forest Regressor
- **Target transform:** Applied `log1p` to the target (`Price`) before training, then `expm1` to convert predictions back to the original scale — this improved model performance on skewed price data
- **Evaluation metric:** R² score
- **Result:** R² ≈ 0.886 on the test set after log transformation

## Project Structure

```
flight-price-prediction/
├── flight_price_prediction.ipynb   # Main notebook: EDA, feature engineering, modeling
├── requirements.txt                 # Python dependencies
├── README.md
└── .gitignore
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/BackpropBoy/flight-price-prediction.git
   cd flight-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook flight_price_prediction.ipynb
   ```

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn (EDA visualizations)
- joblib (model persistence)

## Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Try additional models (XGBoost, Gradient Boosting) for comparison
- Deploy as a simple web app (Flask/Streamlit) for interactive price prediction
