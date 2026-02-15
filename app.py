import streamlit as st

st.set_page_config(page_title="Shopper Spectrum", layout="wide")

st.title("🛒 Shopper Spectrum")
st.markdown("### Customer Segmentation & Product Recommendation")

menu = st.sidebar.selectbox("Choose Module",
                            ["Product Recommendation",
                             "Customer Segmentation"])

if menu == "Product Recommendation":

    st.header("🔎 Product Recommendation System")

    product = st.text_input("Enter Product Name")

    if st.button("Get Recommendations"):
        st.success("Top 5 Recommended Products:")
        st.write("✔ Product A")
        st.write("✔ Product B")
        st.write("✔ Product C")
        st.write("✔ Product D")
        st.write("✔ Product E")

elif menu == "Customer Segmentation":

    st.header("📊 Customer Segmentation")

    recency = st.number_input("Recency (days)", min_value=0)
    frequency = st.number_input("Frequency", min_value=0)
    monetary = st.number_input("Monetary", min_value=0.0)

    if st.button("Predict Customer Segment"):
        st.success("Predicted Segment: High-Value Customer 💎")
