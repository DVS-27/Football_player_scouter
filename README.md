# Football_player_scouter
Football player scouter w/ price
A Python tool that estimates a football player's market value (in € millions) using a Random Forest regression model.

## What it does
- Generates a synthetic dataset of 500 players
- Trains a `RandomForestRegressor` (scikit-learn) on 400 players and tests it on 100
- Evaluates the model using Mean Absolute Error (MAE) and R² score
- Includes a scouting demo that estimates the value of a new young, high-potential player

## Features used
`age`, `overall_rating`, `potential_rating`, `goals`, `assists`, `minutes_played`, `contract_years_left`

**Target:** `market_value_eur_m`

## How to run
```bash
pip install pandas numpy scikit-learn
python football_price_predictor.py
```

## Tech stack
Python, pandas, NumPy, scikit-learn

## Limitations
The dataset is **synthetic**: market values are generated from a formula using the same features the model sees, so the model scores very high (R² close to 1). This shows the pipeline works, but it does not reflect real-world accuracy.

## Next steps
- Train on a real player dataset (e.g. FIFA data from Kaggle)
- Add feature importance analysis
- Compare against simpler models such as Linear Regression

## What I learned
Building a full machine learning workflow: preparing data, splitting it, training a model, evaluating it with MAE and R², and recognising why a perfect score on synthetic data is not meaningful.
