from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import os

def train_model(df, save_path="../outputs/model_performance.txt"):
    """Train linear regression model and save performance metrics."""
    X = df[["Rainfall_mm", "Temperature_C", "Fertilizer_kg"]]
    y = df["Yield_kg_ha"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R2": round(r2_score(y_test, y_pred), 2),
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
    }

    # ensure folder exists before saving
    folder = os.path.abspath(os.path.dirname(save_path))
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    return model, metrics

def predict_yield(model, rainfall, temperature, fertilizer):
    """Make yield prediction using trained model."""
    return model.predict([[rainfall, temperature, fertilizer]])[0]
