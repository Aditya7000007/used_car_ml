import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Used Car Price + Buyer Risk Analyzer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, modern portfolio-grade UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.2rem;
    }
    .badge-low-risk {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-high-risk {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .segment-badge {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Load Models & Cached Assets
# ==============================================================================
@st.cache_resource
def load_ml_assets():
    """Load trained models and preprocessors once and cache in memory."""
    models_dir = "models"
    price_model = joblib.load(os.path.join(models_dir, "price_model.joblib"))
    risk_model = joblib.load(os.path.join(models_dir, "dt_risk_model.joblib"))
    cluster_scaler = joblib.load(os.path.join(models_dir, "cluster_scaler.joblib"))
    kmeans_model = joblib.load(os.path.join(models_dir, "kmeans_model.joblib"))
    nn_recommender = joblib.load(os.path.join(models_dir, "nn_recommender.joblib"))
    
    with open(os.path.join(models_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
        
    cars_df = pd.read_csv(os.path.join(models_dir, "cars_clustered.csv"))
    
    return price_model, risk_model, cluster_scaler, kmeans_model, nn_recommender, metadata, cars_df

try:
    price_model, risk_model, cluster_scaler, kmeans_model, nn_recommender, metadata, cars_df = load_ml_assets()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure models are saved in the `models/` directory.")
    st.stop()


# ==============================================================================
# Helper Functions
# ==============================================================================
def format_inr(amount):
    """Format number into clean Indian Rupees notation."""
    if amount >= 10000000:
        return f"₹{amount / 10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹{amount / 100000:.2f} Lakh"
    else:
        return f"₹{amount:,.0f}"


# ==============================================================================
# Header Section
# ==============================================================================
st.markdown('<div class="main-header">🚗 Used Car Price & Buyer Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'An interactive machine learning platform combining <b>Price Prediction</b> (Random Forest), '
    '<b>Buyer Risk Classification</b> (Decision Tree), and <b>Market Clustering</b> (K-Means)'
    '</div>', 
    unsafe_allow_html=True
)

# ==============================================================================
# Sidebar: Vehicle Input Form
# ==============================================================================
st.sidebar.header("📋 Vehicle Specifications")

# 1. Brand & Cascading Model Selector
brands = metadata["brands"]
default_brand_idx = brands.index("Hyundai") if "Hyundai" in brands else 0
selected_brand = st.sidebar.selectbox("Brand", options=brands, index=default_brand_idx)

available_models = metadata["brand_to_models"].get(selected_brand, [])
selected_model = st.sidebar.selectbox("Model", options=available_models)

# Construct synthetic car_name from brand and model
car_name = f"{selected_brand} {selected_model}"

# 2. Key Vehicle Attributes
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    vehicle_age = st.slider("Vehicle Age (Years)", min_value=1, max_value=25, value=5, step=1)
with col_sb2:
    seats = st.selectbox("Seats", options=[2, 4, 5, 6, 7, 8], index=2)

km_driven = st.sidebar.number_input(
    "Kilometers Driven", 
    min_value=500, 
    max_value=500000, 
    value=45000, 
    step=5000,
    help="Total distance recorded on the odometer"
)

# 3. Mechanical & Technical Specifications
st.sidebar.subheader("⚙️ Mechanical Specs")

col_sb3, col_sb4 = st.sidebar.columns(2)
with col_sb3:
    transmission = st.selectbox("Transmission", options=metadata["transmission_types"])
with col_sb4:
    fuel_type = st.selectbox("Fuel Type", options=metadata["fuel_types"])

seller_type = st.sidebar.selectbox("Seller Type", options=metadata["seller_types"])

col_sb5, col_sb6 = st.sidebar.columns(2)
with col_sb5:
    mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, value=18.5, step=0.5)
with col_sb6:
    engine = st.number_input("Engine (cc)", min_value=600, max_value=6000, value=1197, step=50)

max_power = st.sidebar.slider("Max Power (bhp)", min_value=30.0, max_value=600.0, value=82.0, step=1.0)

analyze_button = st.sidebar.button("⚡ Run Full AI Analysis", use_container_width=True, type="primary")


# ==============================================================================
# Model Inference
# ==============================================================================
# Prepare single row DataFrame matching pipeline features
input_dict = {
    "car_name": car_name,
    "brand": selected_brand,
    "model": selected_model,
    "vehicle_age": vehicle_age,
    "km_driven": km_driven,
    "seller_type": seller_type,
    "fuel_type": fuel_type,
    "transmission_type": transmission,
    "mileage": mileage,
    "engine": engine,
    "max_power": max_power,
    "seats": seats
}
input_df = pd.DataFrame([input_dict])

# 1. Price Prediction
predicted_price = float(price_model.predict(input_df)[0])
lower_bound = max(0, predicted_price * 0.92)
upper_bound = predicted_price * 1.08

# 2. Buyer Risk Classification
risk_pred = int(risk_model.predict(input_df)[0])

# 3. Market Clustering & Similar Cars
cluster_features = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
input_scaled = cluster_scaler.transform(input_df[cluster_features])
cluster_id = int(kmeans_model.predict(input_scaled)[0])
segment_name = metadata["segment_names"].get(str(cluster_id), metadata["segment_names"].get(cluster_id, "Standard Vehicle"))

# Find 5 nearest neighbors
distances, indices = nn_recommender.kneighbors(input_scaled, n_neighbors=5)
similar_cars = cars_df.iloc[indices[0]][[
    "car_name", "brand", "model", "vehicle_age", "km_driven", 
    "mileage", "selling_price", "segment_name"
]].copy()
similar_cars["similarity_score"] = (1 / (1 + distances[0])).round(3)


# ==============================================================================
# Main Dashboard Layout
# ==============================================================================
tab_results, tab_similar, tab_methodology = st.tabs([
    "🎯 Valuation & Risk Analysis", 
    "🔍 Similar Vehicle Listings", 
    "📊 Model Performance & Ethics"
])

# ------------------------------------------------------------------------------
# TAB 1: Valuation & Risk Analysis
# ------------------------------------------------------------------------------
with tab_results:
    st.subheader(f"Vehicle Analysis: {selected_brand} {selected_model}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Estimated Fair Market Value</div>
            <div class="metric-value">{format_inr(predicted_price)}</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">
                Confidence Range: {format_inr(lower_bound)} - {format_inr(upper_bound)}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        risk_badge = '<span class="badge-high-risk">⚠️ Elevated Buyer Risk</span>' if risk_pred == 1 else '<span class="badge-low-risk">✅ Lower Buyer Risk</span>'
        risk_desc = (
            "Vehicle exhibits combined high age (&ge; 8 yrs) and high mileage (&ge; 70k km). Expect elevated maintenance and wear."
            if risk_pred == 1 
            else "Vehicle age and mileage fall within standard operational thresholds for its class."
        )
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Buyer Risk Assessment</div>
            <div style="margin-top: 0.5rem; margin-bottom: 0.4rem;">{risk_badge}</div>
            <div style="font-size: 0.82rem; color: #475569; line-height: 1.35;">
                {risk_desc}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">AI Market Segment (K-Means)</div>
            <div style="margin-top: 0.5rem; margin-bottom: 0.4rem;">
                <span class="segment-badge">Cluster #{cluster_id}: {segment_name}</span>
            </div>
            <div style="font-size: 0.82rem; color: #475569; line-height: 1.35;">
                Categorized based on engine displacement ({engine}cc), {max_power} bhp power output, and {seats} seats.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Key Value Drivers Summary
    st.subheader("💡 Key Valuation Insights")
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("Depreciation Stage", f"{vehicle_age} Years Old", delta=f"-{vehicle_age * 6.5:.1f}% estimated wear", delta_color="inverse")
    with col_k2:
        st.metric("Odometer Status", f"{km_driven:,} km", delta="Average use" if km_driven < 60000 else "High mileage", delta_color="normal" if km_driven < 60000 else "inverse")
    with col_k3:
        st.metric("Fuel Economy", f"{mileage} kmpl", delta="High efficiency" if mileage > 18 else "Standard")
    with col_k4:
        st.metric("Performance Index", f"{max_power} bhp", delta=f"{engine} cc")

    st.info(
        "📌 **Feature Importance Insight**: Our trained Random Forest regressor determined that **Max Power (bhp)**, "
        "**Vehicle Age**, and **Kilometers Driven** have the highest predictive weight in determining a car's market value."
    )


# ------------------------------------------------------------------------------
# TAB 2: Similar Vehicle Listings (Nearest Neighbors)
# ------------------------------------------------------------------------------
with tab_similar:
    st.subheader(f"Top 5 Similar Vehicles in the Marketplace")
    st.write(
        "Using unsupervised **Nearest Neighbors on standardized vehicle specs** "
        "(age, mileage, power, engine capacity, seats), these 5 cars offer the closest real-world match:"
    )
    
    avg_similar_price = similar_cars["selling_price"].mean()
    price_diff = predicted_price - avg_similar_price
    
    st.markdown(f"""
    **Market Context:** The average price of the 5 closest matching listings is **{format_inr(avg_similar_price)}**. 
    Your evaluated vehicle is estimated at **{format_inr(predicted_price)}** 
    ({f"+{format_inr(price_diff)} vs benchmark" if price_diff >= 0 else f"-{format_inr(abs(price_diff))} vs benchmark"}).
    """)
    
    # Display table with formatted columns
    display_df = similar_cars.copy()
    display_df["selling_price"] = display_df["selling_price"].apply(format_inr)
    display_df["km_driven"] = display_df["km_driven"].apply(lambda x: f"{x:,} km")
    display_df["vehicle_age"] = display_df["vehicle_age"].apply(lambda x: f"{x} yrs")
    display_df["mileage"] = display_df["mileage"].apply(lambda x: f"{x} kmpl")
    display_df.rename(columns={
        "car_name": "Car Model",
        "selling_price": "Market Price",
        "vehicle_age": "Age",
        "km_driven": "Usage",
        "mileage": "Efficiency",
        "segment_name": "Market Segment",
        "similarity_score": "Match Confidence"
    }, inplace=True)
    
    st.dataframe(
        display_df[["Car Model", "Market Price", "Age", "Usage", "Efficiency", "Market Segment", "Match Confidence"]], 
        use_container_width=True,
        hide_index=True
    )


# ------------------------------------------------------------------------------
# TAB 3: Model Performance & Transparency
# ------------------------------------------------------------------------------
with tab_methodology:
    st.subheader("📐 Architecture & Engineering Overview")
    
    st.markdown("""
    This project is built as an end-to-end Machine Learning portfolio project demonstrating regression, 
    classification, and unsupervised clustering on **15,244 real-world verified used car records**.
    """)
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("### 1. Price Prediction (Regression)")
        st.markdown("""
        * **Algorithms Tested**: Linear Regression (Baseline), Decision Tree Regressor, Random Forest Regressor.
        * **Production Model**: **Random Forest Regressor** (100 estimators, max depth 20).
        * **Test Performance**:
          * **$R^2$ Score**: ~0.895 (explains ~90% of price variance)
          * **MAE**: ≈ ₹98,628
          * **RMSE**: ≈ ₹252,117
        """)
        
        st.markdown("### 2. Market Segmentation (Clustering)")
        st.markdown("""
        * **Algorithm**: **K-Means Clustering** ($K=4$) with `StandardScaler`.
        * **Optimal $K$**: Selected via Elbow Method (Inertia curve inflection at $K=4$).
        * **Cluster Profiles**:
          * `Cluster 0`: Modern Daily Commuters (Hatchbacks/Subcompacts)
          * `Cluster 1`: Luxury & High-Performance Sedans/Coupes
          * `Cluster 2`: 7-Seater Family SUVs & MPVs
          * `Cluster 3`: Budget & Entry-Level Used Cars
        """)
        
    with col_m2:
        st.markdown("### 3. Buyer Risk (Classification)")
        st.markdown("""
        * **Target Definition**: Project-defined heuristic rule:
          $$\\text{High Risk} = (\\text{vehicle\\_age} \\ge 8) \\land (\\text{km\\_driven} \\ge 70{,}000)$$
        * **Model Comparison**:
        """)
        
        comparison_table = pd.DataFrame({
            "Metric": ["Accuracy", "Precision (Class 1)", "Recall (Class 1)", "F1-Score", "Decision Boundary"],
            "Logistic Regression": ["91.54%", "75.98%", "58.70%", "0.6623", "Linear Hyperplane (Diagonal)"],
            "Decision Tree (Production)": ["100.0%", "100.0%", "100.0%", "1.0000", "Orthogonal Thresholds (Exact)"]
        })
        st.table(comparison_table)
        
        st.warning(
            "⚠️ **Methodology & Transparency Notice**: The Buyer Risk score represents a synthetic heuristic rule "
            "designed to showcase classification under class imbalance (85.9% vs 14.1%) and contrast linear vs tree-based "
            "decision boundaries. It is **not** an officially certified mechanical, financial, or warranty guarantee."
        )

# ==============================================================================
# Footer
# ==============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "Used Car Price + Buyer Risk Analyzer | Portfolio Machine Learning Project | Built with Scikit-Learn & Streamlit"
    "</div>", 
    unsafe_allow_html=True
)

