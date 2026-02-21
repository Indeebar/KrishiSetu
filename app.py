import streamlit as st
import pandas as pd
from src.utils.inference import (
    load_cnn_model, 
    predict_image_class, 
    load_rf_model, 
    predict_price,
    CLASS_NAMES
)

# App Configuration
st.set_page_config(
    page_title="KrishiSetu | Agri-Waste Valuator",
    page_icon="🌾",
    layout="centered"
)

# Hide Streamlit menu for a cleaner look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# List of states extracted from the dataset
STATES = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
    'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya',
    'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim',
    'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand',
    'West Bengal'
]

# Load Models (cached so they only load once)
@st.cache_resource
def init_models():
    cnn = load_cnn_model()
    rf = load_rf_model()
    return cnn, rf

# Header
st.title("🌾 KrishiSetu")
st.subheader("Agricultural Waste Valuation System")

st.markdown("""
    Welcome to KrishiSetu! Upload an image of your agricultural waste, 
    select your state, and our AI models will identify the waste type 
    and estimate its current market price.
""")

st.divider()

try:
    cnn_model, rf_pipeline = init_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Models not found. Please train models first using `src/models/train_cnn.py` and `src/models/train_rf.py`.")
    st.info(f"Details: {e}")

# Main Layout
col1, col2 = st.columns([1, 1])

# Sidebar inputs
with st.sidebar:
    st.header("1. Location")
    state_selected = st.selectbox("Select your State:", STATES)
    
    st.header("2. Image Upload")
    uploaded_file = st.file_uploader(
        "Upload an image of the waste (JPG/PNG)", 
        type=["jpg", "jpeg", "png"]
    )
    
    analyze_button_clicked = st.button("Analyze Image", type="primary", use_container_width=True)

# Main Display Area
if uploaded_file is not None:
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
    if analyze_button_clicked and models_loaded:
        with col2:
            with st.spinner('Analyzing...'):
                # 1. Image Classification
                try:
                    predicted_class, confidence = predict_image_class(uploaded_file, cnn_model)
                    
                    st.success("Image Analyzed!")
                    st.metric("Waste Type Detected", predicted_class)
                    st.progress(confidence / 100.0, text=f"Confidence: {confidence:.2f}%")
                    
                    # 2. Price Prediction
                    predicted_price = predict_price(state_selected, predicted_class, rf_pipeline)
                    
                    st.divider()
                    if predicted_price is not None:
                        st.success("Price Estimated!")
                        st.metric("Estimated Market Price", f"₹{predicted_price:.2f} per kg")
                        st.info("Note: Price is an estimate based on recent market data for your region.")
                    else:
                        st.warning("Price Estimation Unavailable")
                        st.info(f"We don't have enough market data to estimate the price for '{predicted_class}' in your region yet.")
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
elif analyze_button_clicked and uploaded_file is None:
    st.warning("Please upload an image first.")

