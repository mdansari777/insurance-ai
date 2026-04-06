import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from .preprocessing import get_health_preprocessor, get_car_preprocessor, get_life_preprocessor, get_home_preprocessor

def train_regression(df, target_col, preprocessor, model_name, param_grids):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, verbosity=0)
    }

    best_estimators = {}
    results = []

    for name, model in models.items():
        print(f"Training {name} for {model_name}...")
        pipe = Pipeline([('preprocessor', preprocessor), ('regressor', model)])
        param_grid = param_grids.get(name, {})
        if param_grid:
            gs = GridSearchCV(pipe, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)
            best_pipe = gs.best_estimator_
        else:
            pipe.fit(X_train, y_train)
            best_pipe = pipe

        y_pred = best_pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({'model': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2})
        best_estimators[name] = best_pipe

    best_model_name = max(results, key=lambda x: x['R2'])['model']
    best_model = best_estimators[best_model_name]

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f'models/{model_name}_best.pkl')
    pd.DataFrame(results).to_csv(f'models/{model_name}_regression_results.csv', index=False)
    print(f"✅ Best model for {model_name}: {best_model_name} (R2={max(r['R2'] for r in results):.3f})")

def train_all_models():
    print("="*50)
    print("Training Health Model...")
    df = pd.read_csv('data/health_insurance.csv')
    preprocessor = get_health_preprocessor()
    param_grids = {
        'RandomForest': {'regressor__n_estimators': [50,100], 'regressor__max_depth': [5,10]},
        'GradientBoosting': {'regressor__n_estimators': [50,100], 'regressor__learning_rate': [0.05,0.1]},
        'XGBoost': {'regressor__n_estimators': [50,100], 'regressor__learning_rate': [0.05,0.1]}
    }
    train_regression(df, 'charges', preprocessor, 'health', param_grids)

    print("\n" + "="*50)
    print("Training Car Model...")
    df = pd.read_csv('data/car_insurance.csv')
    preprocessor = get_car_preprocessor()
    train_regression(df, 'premium', preprocessor, 'car', param_grids)

    print("\n" + "="*50)
    print("Training Life Model...")
    df = pd.read_csv('data/life_insurance.csv')
    preprocessor = get_life_preprocessor()
    train_regression(df, 'premium', preprocessor, 'life', param_grids)

    print("\n" + "="*50)
    print("Training Home Model...")
    df = pd.read_csv('data/home_insurance.csv')
    preprocessor = get_home_preprocessor()
    train_regression(df, 'premium', preprocessor, 'home', param_grids)

    print("\n✅ All regression models trained successfully!")

if __name__ == "__main__":
    train_all_models()