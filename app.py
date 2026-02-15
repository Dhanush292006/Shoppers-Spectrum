import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Shopper Spectrum", layout="wide")

st.title("🛒 Shopper Spectrum")
st.markdown("### Customer Segmentation & Product Recommendation")

# --------------------------------------------------
# LOAD DATA AND MODELS
# --------------------------------------------------

@st.cache_resource
def load_all():
    
    # Load small dataset
    df = pd.read_csv("online_retail_small.csv", encoding='ISO-8859-1')

    # Cleaning
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Build pivot
    pivot = df.pivot_table(index='CustomerID',
                           columns='Description',
                           values='Quantity',
                           fill_value=0)

    # Build similarity matrix
    similarity = cosine_similarity(pivot.T)

    # Load clustering model
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")

    return pivot, similarity, kmeans, scaler

pivot, item_similarity, kmeans, scaler = load_all()

# --------------------------------------------------
# SIDEBAR MENU
# --------------------------------------------------

menu = st.sidebar.selectbox(
    "Choose Module",
    ["Product Recommendation", "Customer Segmentation"]
)

# --------------------------------------------------
# PRODUCT RECOMMENDATION
# --------------------------------------------------

if menu == "Product Recommendation":

    st.header("🔎 Product Recommendation System")

    product = st.text_input("Enter Exact Product Name")

    if st.button("Get Recommendations"):

        if product not in pivot.columns:
            st.error("❌ Product not found. Please check exact spelling.")
        else:
            index = list(pivot.columns).index(product)

            similarity_scores = list(enumerate(item_similarity[index]))
            similarity_scores = sorted(
                similarity_scores,
                key=lambda x: x[1],
                reverse=True
            )[1:6]

            st.success("Top 5 Recommended Products:")

            for i in similarity_scores:
                st.write("✔", pivot.columns[i[0]])

# --------------------------------------------------
# CUSTOMER SEGMENTATION
# --------------------------------------------------

elif menu == "Customer Segmentation":

    st.header("📊 Customer Segmentation (RFM Based)")

    recency = st.number_input("Recency (days)", min_value=0)
    frequency = st.number_input("Frequency", min_value=0)
    monetary = st.number_input("Monetary", min_value=0.0)

    if st.button("Predict Customer Segment"):

        input_data = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(input_data)[0]

        segment_labels = {
            0: "💎 High-Value Customer",
            1: "🙂 Regular Customer",
            2: "🛍 Occasional Customer",
            3: "⚠ At-Risk Customer"
        }

        st.success(f"Predicted Segment: {segment_labels.get(cluster)}")
