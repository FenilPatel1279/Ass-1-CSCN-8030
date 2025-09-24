import pandas as pd
import os

def load_data(path="../data/crop_yield_sample.csv"):
    """Load dataset from CSV if available, else generate synthetic."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not df.empty:
            return df
    return generate_sample_data(path)

def save_data(df, path="../data/crop_yield_sample.csv"):
    """Save dataframe as CSV (ensures folder exists)."""
    folder = os.path.abspath(os.path.dirname(path))
    os.makedirs(folder, exist_ok=True)
    df.to_csv(path, index=False)

def generate_sample_data(path="../data/crop_yield_sample.csv"):
    """Generate synthetic dataset if file missing/empty."""
    data = {
        "Rainfall_mm": [350, 420, 500, 600, 700, 300, 450, 550],
        "Temperature_C": [18, 20, 22, 25, 27, 30, 24, 26],
        "Fertilizer_kg": [80, 100, 110, 120, 130, 90, 115, 125],
        "Yield_kg_ha": [2800, 3200, 3500, 3800, 3700, 2600, 3300, 3600]
    }
    df = pd.DataFrame(data)
    save_data(df, path)
    return df

