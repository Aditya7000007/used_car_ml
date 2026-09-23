"""
Unit tests for data preprocessing, model inference, and recommendation pipelines.
"""

import unittest
import os
import pandas as pd
from src.data_preprocessing import load_clean_data, add_buyer_risk_target
from src.model_predictor import CarPriceRiskPredictor
from src.recommender import CarRecommender


class TestUsedCarMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = CarPriceRiskPredictor(models_dir="models")
        cls.recommender = CarRecommender(models_dir="models")

        cls.sample_car = {
            "car_name": "Maruti Swift",
            "brand": "Maruti",
            "model": "Swift",
            "vehicle_age": 4,
            "km_driven": 35000,
            "seller_type": "Individual",
            "fuel_type": "Petrol",
            "transmission_type": "Manual",
            "mileage": 21.2,
            "engine": 1197,
            "max_power": 81.8,
            "seats": 5
        }

    def test_data_cleaning(self):
        """Test dataset loading and seat median imputation."""
        df = load_clean_data("data/cardekho_dataset.csv")
        self.assertNotIn("Unnamed: 0", df.columns)
        self.assertEqual(df["seats"].min(), 2)  # Seats == 0 was imputed
        self.assertEqual(len(df), 15244)

    def test_buyer_risk_target(self):
        """Test synthetic buyer risk rule."""
        df = pd.DataFrame({
            "vehicle_age": [9, 7, 10, 4],
            "km_driven": [80000, 90000, 50000, 20000]
        })
        df_risk = add_buyer_risk_target(df)
        self.assertEqual(df_risk["buyer_risk"].tolist(), [1, 0, 0, 0])

    def test_price_prediction(self):
        """Test price prediction returns positive and valid bounds."""
        price, low, high = self.predictor.predict_price(self.sample_car)
        self.assertGreater(price, 100000)
        self.assertLess(price, 30000000)
        self.assertLess(low, price)
        self.assertGreater(high, price)

    def test_risk_prediction(self):
        """Test risk classifier outputs 0 for a young car with low mileage."""
        risk_class, risk_label = self.predictor.predict_risk(self.sample_car)
        self.assertEqual(risk_class, 0)
        self.assertEqual(risk_label, "Lower Buyer Risk")

    def test_recommender(self):
        """Test recommendation returns 5 similar vehicles."""
        similar = self.recommender.recommend_similar(self.sample_car, top_n=5)
        self.assertEqual(len(similar), 5)
        self.assertIn("car_name", similar.columns)
        self.assertIn("selling_price", similar.columns)


if __name__ == "__main__":
    unittest.main()

