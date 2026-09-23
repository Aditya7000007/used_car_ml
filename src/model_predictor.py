"""
Inference & Prediction Service Module
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any, Tuple


class CarPriceRiskPredictor:
    """
    Unified predictor service for both used car price estimation and buyer risk classification.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.price_model = joblib.load(os.path.join(models_dir, "price_model.joblib"))
        self.risk_model = joblib.load(os.path.join(models_dir, "dt_risk_model.joblib"))

    def predict_price(self, car_data: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Estimates resale price along with a +/- 8% confidence interval.
        """
        df = pd.DataFrame([car_data])
        price = float(self.price_model.predict(df)[0])
        lower_bound = max(0.0, price * 0.92)
        upper_bound = price * 1.08
        return price, lower_bound, upper_bound

    def predict_risk(self, car_data: Dict[str, Any]) -> Tuple[int, str]:
        """
        Classifies buyer risk:
        0 -> Lower Buyer Risk
        1 -> Elevated Buyer Risk
        """
        df = pd.DataFrame([car_data])
        risk_class = int(self.risk_model.predict(df)[0])
        risk_label = "Elevated Buyer Risk" if risk_class == 1 else "Lower Buyer Risk"
        return risk_class, risk_label

