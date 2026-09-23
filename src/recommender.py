"""
Market Clustering & Similar Car Recommender Engine
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


class CarRecommender:
    """
    Encapsulates K-Means market segmentation and Nearest Neighbors similarity search.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.scaler = joblib.load(os.path.join(models_dir, "cluster_scaler.joblib"))
        self.kmeans = joblib.load(os.path.join(models_dir, "kmeans_model.joblib"))
        self.nn_model = joblib.load(os.path.join(models_dir, "nn_recommender.joblib"))
        self.cars_df = pd.read_csv(os.path.join(models_dir, "cars_clustered.csv"))
        
        self.features = [
            "vehicle_age", "km_driven", "mileage", 
            "engine", "max_power", "seats"
        ]
        
        self.segment_names = {
            0: "Modern Daily Commuter (Hatchback/Subcompact)",
            1: "Luxury & High-Performance Vehicle",
            2: "7-Seater Family SUV / MPV",
            3: "Budget & Entry-Level Used Car"
        }

    def predict_segment(self, car_specs: Dict[str, Any]) -> Tuple[int, str]:
        """
        Assigns the vehicle to one of 4 K-Means market clusters.
        """
        input_df = pd.DataFrame([car_specs])[self.features]
        scaled_input = self.scaler.transform(input_df)
        cluster_id = int(self.kmeans.predict(scaled_input)[0])
        segment_name = self.segment_names.get(cluster_id, "Standard Vehicle")
        return cluster_id, segment_name

    def recommend_similar(self, car_specs: Dict[str, Any], top_n: int = 5) -> pd.DataFrame:
        """
        Finds the top_n nearest vehicles in the marketplace dataset matching given specs.
        """
        input_df = pd.DataFrame([car_specs])[self.features]
        scaled_input = self.scaler.transform(input_df)
        distances, indices = self.nn_model.kneighbors(scaled_input, n_neighbors=top_n)

        results = self.cars_df.iloc[indices[0]][[
            "car_name", "brand", "model", "selling_price", 
            "vehicle_age", "km_driven", "mileage", "segment_name"
        ]].copy()

        # Compute match confidence score based on inverse euclidean distance
        results["match_confidence"] = (1.0 / (1.0 + distances[0])).round(3)
        results["euclidean_distance"] = distances[0].round(3)
        return results

