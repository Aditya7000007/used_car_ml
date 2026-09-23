"""
Data Preprocessing & Feature Engineering Module
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLUMNS = [
    "car_name",
    "brand",
    "model",
    "seller_type",
    "fuel_type",
    "transmission_type"
]

NUMERICAL_COLUMNS = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

CLUSTER_FEATURES = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]


def load_clean_data(csv_path: str = "data/cardekho_dataset.csv") -> pd.DataFrame:
    """
    Loads raw Cardekho dataset, removes unwanted index columns,
    drops exact duplicates, and imputes invalid seat values using median.
    """
    if not os.path.exists(csv_path):
        # Fallback check for relative paths
        alt_path = os.path.join("..", csv_path)
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Drop index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 2. Drop duplicate records
    df = df.drop_duplicates()

    # 3. Handle invalid seats (seats == 0)
    df.loc[df["seats"] == 0, "seats"] = np.nan
    df["seats"] = df["seats"].fillna(df["seats"].median())

    return df


def add_buyer_risk_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends the project-defined heuristic buyer risk classification target:
    0 = Lower Risk, 1 = Elevated Risk (Vehicle Age >= 8 years AND KM Driven >= 70,000 km).
    """
    df = df.copy()
    df["buyer_risk"] = (
        (df["vehicle_age"] >= 8) & 
        (df["km_driven"] >= 70000)
    ).astype(int)
    return df


def get_preprocessor() -> ColumnTransformer:
    """
    Constructs the standard ColumnTransformer with numerical passthrough
    and categorical OneHotEncoding (ignoring unknown categories at test time).
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
        ]
    )

