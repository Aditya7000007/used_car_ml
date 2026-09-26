"""Standalone website server for the used-car analysis project (Render-ready)."""

import json
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

app = Flask(__name__, template_folder="templates", static_folder="static")

# Allow Firebase Hosting + localhost for local dev
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "https://*.web.app",
    "https://*.firebaseapp.com",
]}})


def load_assets():
    """Load the project's saved models and listing data once at startup."""
    with (MODELS_DIR / "metadata.json").open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    assets = {
        "price_model": joblib.load(MODELS_DIR / "price_model.joblib"),
        "risk_model": joblib.load(MODELS_DIR / "dt_risk_model.joblib"),
        "scaler": joblib.load(MODELS_DIR / "cluster_scaler.joblib"),
        "kmeans": joblib.load(MODELS_DIR / "kmeans_model.joblib"),
        "neighbors": joblib.load(MODELS_DIR / "nn_recommender.joblib"),
        "cars": pd.read_csv(MODELS_DIR / "cars_clustered.csv"),
        "metadata": metadata,
    }
    return assets


ASSETS = load_assets()
CLUSTER_FEATURES = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
NUMERIC_FEATURES = CLUSTER_FEATURES
INTEGER_FEATURES = {"vehicle_age", "km_driven", "engine", "seats"}


def validate_input(payload):
    """Validate user input and return the model-ready dictionary."""
    if not isinstance(payload, dict):
        raise ValueError("Send vehicle details as a JSON object.")

    metadata = ASSETS["metadata"]
    brand = payload.get("brand")
    model = payload.get("model")
    if brand not in metadata["brands"]:
        raise ValueError("Choose a listed vehicle brand.")
    if model not in metadata["brand_to_models"].get(brand, []):
        raise ValueError("Choose a model listed for that brand.")

    categories = {
        "seller_type": "seller_types",
        "fuel_type": "fuel_types",
        "transmission_type": "transmission_types",
    }
    for field, metadata_key in categories.items():
        if payload.get(field) not in metadata[metadata_key]:
            raise ValueError(f"Choose a valid {field.replace('_', ' ')}.")

    car = {
        "brand": brand,
        "model": model,
        "car_name": f"{brand} {model}",
        "seller_type": payload["seller_type"],
        "fuel_type": payload["fuel_type"],
        "transmission_type": payload["transmission_type"],
    }

    for field in NUMERIC_FEATURES:
        raw_value = payload.get(field)
        if isinstance(raw_value, bool):
            raise ValueError(f"Enter a number for {field.replace('_', ' ')}.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"Enter a number for {field.replace('_', ' ')}.") from None
        if not math.isfinite(value):
            raise ValueError(f"Enter a finite number for {field.replace('_', ' ')}.")
        if field in INTEGER_FEATURES and not value.is_integer():
            raise ValueError(f"{field.replace('_', ' ').title()} must be a whole number.")

        bounds = metadata["feature_ranges"][field]
        if value < bounds["min"] or value > bounds["max"]:
            raise ValueError(
                f"{field.replace('_', ' ').title()} must be between "
                f"{bounds['min']:g} and {bounds['max']:g}."
            )
        car[field] = int(value) if field in INTEGER_FEATURES else value

    return car


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/metadata")
def get_metadata():
    metadata = ASSETS["metadata"]
    return jsonify({
        "brands": metadata["brands"],
        "brand_to_models": metadata["brand_to_models"],
        "seller_types": metadata["seller_types"],
        "fuel_types": metadata["fuel_types"],
        "transmission_types": metadata["transmission_types"],
        "feature_ranges": metadata["feature_ranges"],
    })


@app.post("/api/analyze")
def analyze():
    try:
        car = validate_input(request.get_json(silent=True))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        car_frame = pd.DataFrame([car])
        price = max(0.0, float(ASSETS["price_model"].predict(car_frame)[0]))
        risk_flag = int(ASSETS["risk_model"].predict(car_frame)[0])

        feature_frame = car_frame[CLUSTER_FEATURES]
        scaled_features = ASSETS["scaler"].transform(feature_frame)
        cluster_id = int(ASSETS["kmeans"].predict(scaled_features)[0])
        segment = ASSETS["metadata"]["segment_names"].get(str(cluster_id), "Standard Vehicle")

        count = min(5, len(ASSETS["cars"]))
        distances, indices = ASSETS["neighbors"].kneighbors(scaled_features, n_neighbors=count)
        listings = []
        for distance, row_index in zip(distances[0], indices[0]):
            row = ASSETS["cars"].iloc[int(row_index)]
            listings.append({
                "car_name": str(row["car_name"]),
                "brand": str(row["brand"]),
                "model": str(row["model"]),
                "selling_price": float(row["selling_price"]),
                "vehicle_age": int(row["vehicle_age"]),
                "km_driven": int(row["km_driven"]),
                "mileage": float(row["mileage"]),
                "segment_name": str(row["segment_name"]),
                "distance_similarity": round(1 / (1 + float(distance)), 3),
            })

        avg_listing_price = sum(item["selling_price"] for item in listings) / len(listings)
        return jsonify({
            "price": price,
            "price_range": {"low": max(0.0, price * 0.92), "high": price * 1.08},
            "screening_flag": risk_flag,
            "screening_rule_met": car["vehicle_age"] >= 8 and car["km_driven"] >= 70000,
            "cluster_id": cluster_id,
            "segment": segment,
            "average_listing_price": avg_listing_price,
            "price_difference": price - avg_listing_price,
            "similar_cars": listings,
        })
    except Exception:
        app.logger.exception("Vehicle analysis failed")
        return jsonify({"error": "Analysis failed. Check the model files and try again."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
