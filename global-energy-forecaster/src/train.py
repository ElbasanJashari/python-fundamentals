from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# 1. LOAD
DATA = Path(__file__).parent.parent / "data" / "PJME_hourly.csv"
df = pd.read_csv(DATA, parse_dates=['Datetime'])
df = df.sort_values('Datetime').set_index('Datetime')
df.columns = ['MW']  # target


# 2. FEATURE ENGINEERING
def make_features(data, lags=[24, 168], window=24):
    df = data.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['MW'].shift(lag)
    df[f'roll_mean_{window}'] = df['MW'].shift(1).rolling(window).mean()
    df[f'roll_std_{window}'] = df['MW'].shift(1).rolling(window).std()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df.dropna()


feat_df = make_features(df)
X = feat_df.drop(columns=['MW'])
y = feat_df['MW']


# 3. TIME-BASED SPLIT
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


# 4. PIPELINE
def get_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


pipe_ridge = Pipeline([
    ("pre", get_preprocessor()),
    ("model", Ridge())
])

pipe_rf = Pipeline([
    ("pre", get_preprocessor()),
    ("model", RandomForestRegressor(n_estimators=100,
     max_depth=10, random_state=42, n_jobs=-1))
])

# 5. BASELINE vs MODEL
cv = TimeSeriesSplit(n_splits=5)

# naive baseline: predict lag_24
naive_pred = X_test['lag_24']
print(f"Naive MAE: {mean_absolute_error(y_test, naive_pred):.2f}")

# Ridge
pipe_ridge.fit(X_train, y_train)
pred_ridge = pipe_ridge.predict(X_test)
print(f"Ridge MAE: {mean_absolute_error(y_test, pred_ridge):.2f} | RMSE: {np.sqrt(mean_squared_error(y_test, pred_ridge)):.2f} | R2: {r2_score(y_test, pred_ridge):.3f}")

# RF + GridSearch
param_grid = {"model__max_depth": [10, 20], "model__n_estimators": [100, 200]}
grid = GridSearchCV(pipe_rf, param_grid, cv=cv,
                    scoring='neg_mean_absolute_error', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best: {grid.best_params_} | Best CV MAE: {-grid.best_score_:.2f}")
final_pred = grid.predict(X_test)
print(f"Final Test MAE: {mean_absolute_error(y_test, final_pred):.2f} | RMSE: {np.sqrt(mean_squared_error(y_test, final_pred)):.2f} | R2: {r2_score(y_test, final_pred):.3f}")


Path("models").mkdir(exist_ok=True)
joblib.dump(grid.best_estimator_, "models/rf_energy_model.pkl")
print("Model saved to models/rf_energy_model.pkl")
