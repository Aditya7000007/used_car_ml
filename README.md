# 🚗 Used Car Price + Buyer Risk Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning portfolio application designed to solve information asymmetry in the pre-owned vehicle marketplace. The system combines **Regression** for fair market valuation, **Classification** for buyer-risk detection, and **Unsupervised Clustering** for market segmentation and similar-car recommendations.

---

## 📌 Project Highlights

- **Regression (Price Prediction):** Evaluated Linear Regression, Decision Trees, and **Random Forest Regressor** ($R^2 \approx 0.895$, $\text{MAE} \approx ₹98{,}628$).
- **Classification (Buyer Risk):** Evaluated Logistic Regression vs. **Decision Tree Classifier** under an 86/14 class imbalance, highlighting the difference between linear and orthogonal decision boundaries.
- **Unsupervised Learning (Market Segmentation):** Applied **K-Means Clustering** ($K=4$) with `StandardScaler` and the Elbow Method to segment cars into intuitive market tiers.
- **Similar Car Recommender:** Deployed a **Nearest Neighbors** Euclidean distance search to match any vehicle input against 15,244 real marketplace listings.
- **Interactive Web Dashboard:** Modern, responsive **Streamlit** dashboard featuring dynamic cascading dropdowns, instant valuation, risk assessment, and live benchmark comparisons.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Raw Data: 15,411 Records] --> B[Data Cleaning & Deduplication: 15,244 Records]
    B --> C[Feature Engineering & Preprocessing]
    
    C --> D[Supervised: Price Regression]
    C --> E[Supervised: Buyer Risk Classification]
    C --> F[Unsupervised: Market Clustering]
    
    D --> D1[Random Forest Regressor<br/>R²: 0.895 | MAE: ₹98.6k]
    E --> E1[Decision Tree vs Logistic Regression<br/>Class Imbalance Evaluation]
    F --> F1[K-Means Clustering K=4<br/>& Nearest Neighbors Recommender]
    
    D1 --> G[Streamlit Dashboard: app.py]
    E1 --> G
    F1 --> G
```

---

## 📊 Dataset & Preprocessing

- **Source:** CarDekho Used Car Dataset (`cardekho_dataset.csv`)
- **Cleaning:**
  - Removed 167 duplicate records ($15,411 \to 15,244$ unique rows).
  - Imputed invalid zero-seat entries (`seats == 0`) using the dataset median ($5$).
  - **Outlier Philosophy:** Luxury and exotic vehicles (e.g., Ferrari GTC4Lusso, Rolls-Royce Ghost, Bentley Continental) were **intentionally retained** rather than clipped, preserving natural market variance for high-end vehicles.
- **Preprocessing Pipeline (`ColumnTransformer`):**
  - **Numerical Features** (`vehicle_age`, `km_driven`, `mileage`, `engine`, `max_power`, `seats`): Standardized or passed through depending on downstream model requirements.
  - **Categorical Features** (`car_name`, `brand`, `model`, `seller_type`, `fuel_type`, `transmission_type`): Encoded using `OneHotEncoder(handle_unknown="ignore")`.

---

## 📈 Machine Learning Models & Results

### 1. Used Car Price Prediction (Regression)

| Model | MAE | RMSE | $R^2$ Score | Notes |
|---|---|---|---|---|
| **Linear Regression** | ₹244,529 | ₹427,308 | 0.700 | Baseline model |
| **Decision Tree Regressor** | ₹121,948 | ₹309,120 | 0.843 | `max_depth=20` |
| **Random Forest Regressor** ⭐ | **₹98,628** | **₹252,117** | **0.895** | **Production model** (`n_estimators=100`, `max_depth=20`) |

**Feature Importance Analysis:**  
Random Forest feature importance showed that **Max Power (bhp)**, **Vehicle Age**, and **Kilometers Driven** are the top 3 drivers of resale valuation.

---

### 2. Buyer Risk Classification

To explore classification under real-world class imbalance, a target label was formulated using upper-quartile operational thresholds:
$$\text{High Risk} = (\text{vehicle\_age} \ge 8\text{ years}) \land (\text{km\_driven} \ge 70{,}000\text{ km})$$

- **Class Distribution:** 85.9% Class 0 (Lower Risk) vs. 14.1% Class 1 (Elevated Risk)

| Metric | Logistic Regression | Decision Tree Classifier ⭐ |
|---|---|---|
| **Accuracy** | 91.54% | **100.0%** |
| **Precision (Class 1)** | 75.98% | **100.0%** |
| **Recall (Class 1)** | 58.70% (missed 178 risky cars) | **100.0%** |
| **F1-Score (Class 1)** | 66.23% | **100.0%** |
| **Decision Boundary** | Linear Hyperplane (diagonal) | Orthogonal Thresholds (axis-aligned) |

> **⚠️ Transparency Note:** The buyer risk target serves as a pedagogical benchmark to evaluate linear vs tree-based models under class imbalance. In production, this heuristic highlights vehicles with statistically higher probability of wear, but does not substitute for a physical mechanical inspection.

---

### 3. Market Clustering & Similar Cars (Unsupervised)

Using the **Elbow Method** on normalized vehicle specs (`StandardScaler`), $K=4$ was selected:

| Cluster | Segment Name | Avg Age | Avg KM | Avg Engine | Avg Price | Typical Models |
|---|---|---|---|---|---|---|
| **0** | Modern Daily Commuters | 4.2 yrs | 41k km | 1180 cc | ₹6.2 Lakh | Swift, i20, Baleno, Amaze |
| **1** | Luxury & High-Performance | 6.7 yrs | 53k km | 2300 cc | ₹23.7 Lakh | BMW 3/5, Mercedes C/E, Audi A4/A6 |
| **2** | 7-Seater Family SUVs & MPVs | 6.4 yrs | 83k km | 2100 cc | ₹11.2 Lakh | Innova, Ertiga, Scorpio, XUV500 |
| **3** | Budget & Entry-Level Used Cars | 9.3 yrs | 73k km | 1200 cc | ₹3.5 Lakh | Alto, WagonR, Santro |

---

## 🗂️ Project Structure

```text
used_car_ml/
├── data/
│   └── cardekho_dataset.csv       # Cleaned used car dataset
├── models/
│   ├── price_model.joblib         # Production Random Forest Regressor
│   ├── dt_risk_model.joblib       # Production Decision Tree Classifier
│   ├── cluster_scaler.joblib      # StandardScaler for clustering features
│   ├── kmeans_model.joblib        # 4-Cluster KMeans model
│   ├── nn_recommender.joblib      # NearestNeighbors recommender model
│   ├── cars_clustered.csv         # Database with cluster assignments
│   └── metadata.json              # Dynamic UI configurations & feature bounds
├── notebook/
│   └── 01_data_exploration.ipynb  # End-to-end EDA, experiments & evaluation
├── app.py                         # Streamlit interactive web application
├── requirements.txt               # Production Python dependencies
└── README.md                      # Documentation & portfolio presentation
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/used_car_ml.git
cd used_car_ml
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit application
```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn (Random Forest, Decision Trees, Logistic Regression, K-Means, Nearest Neighbors)
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Web App / UI:** Streamlit
- **Model Serialization:** Joblib

---

## 👤 Author

- **Aditya**
- Portfolio Project: Used Car Price & Buyer Risk Analyzer
- Target: Machine Learning Engineering & Data Science

