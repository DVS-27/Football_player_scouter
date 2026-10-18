import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def create_mock_football_dataset():
    np.random.seed(42)
    n_samples = 500
    
    ages = np.random.randint(18, 38, size=n_samples)
    overall_ratings = np.random.randint(60, 95, size=n_samples)
    potential_ratings = overall_ratings + np.random.randint(0, 10, size=n_samples)
    potential_ratings = np.clip(potential_ratings, 0, 99)
    
    goals = np.random.poisson(lam=5, size=n_samples)
    assists = np.random.poisson(lam=4, size=n_samples)
    minutes_played = np.random.randint(500, 3000, size=n_samples)
    contract_years_left = np.random.randint(0, 5, size=n_samples)
    
    base_value = (overall_ratings ** 2.2) * 0.05
    age_multiplier = np.where(ages < 28, 1.2, 0.7)
    market_value = (base_value + (goals * 2.5) + (assists * 2.0) + (contract_years_left * 3)) * age_multiplier
    market_value = np.round(np.clip(market_value, 0.5, 150.0), 2)

    df = pd.DataFrame({
        'age': ages,
        'overall_rating': overall_ratings,
        'potential_rating': potential_ratings,
        'goals': goals,
        'assists': assists,
        'minutes_played': minutes_played,
        'contract_years_left': contract_years_left,
        'market_value_eur_m': market_value
    })
    return df

def main():
    print("--- 1. Generating & Loading Football Dataset ---")
    df = create_mock_football_dataset()
    print(df.head(), "\n")

    X = df[['age', 'overall_rating', 'potential_rating', 'goals', 'assists', 'minutes_played', 'contract_years_left']]
    y = df['market_value_eur_m']

    print("--- 3. Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}\n")

    print("--- 4. Training the Regressor Model ---")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete!\n")

    print("--- 5. Evaluating the Model ---")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error: €{mae:.2f} Million")
    print(f"R² Score: {r2:.2f}\n")

    print("--- 6. Scouting a New Player ---")
    wonderkid = [[21, 82, 91, 14, 8, 2400, 4]] 
    predicted_price = model.predict(wonderkid)
    
    print("Scouted Player Stats:")
    print(" - Age: 21 | Overall: 82 | Potential: 91 | Goals: 14 | Assists: 8")
    print(f"Estimated Market Value: **€{predicted_price[0]:.2f} Million** 💰")

if __name__ == "__main__":
    main()