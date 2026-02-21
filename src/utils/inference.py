import os
import io
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# Global paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "../../models")
CNN_MODEL_PATH = os.path.abspath(os.path.join(MODELS_DIR, "custom_cnn_model.keras"))
RF_MODEL_PATH = os.path.abspath(os.path.join(MODELS_DIR, "price_predictor.pkl"))

# Note: Dynamically loading class names from the dataset directory to support retraining
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../Agri_Waste_Images"))
try:
    CLASS_NAMES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
except FileNotFoundError:
    CLASS_NAMES = [] # Fallback if directory isn't mounted the same way

# --- CNN Inference (Image Classification) ---

def load_cnn_model():
    """Loads the trained Keras CNN model."""
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"CNN model not found at {CNN_MODEL_PATH}. Did you run train_cnn.py?")
    
    print("Loading CNN model...")
    model = tf.keras.models.load_model(CNN_MODEL_PATH)
    return model

def predict_image_class(uploaded_file, model):
    """Predicts the class of the uploaded image."""
    # Read the image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Resize to match training data (224x224)
    image = image.resize((224, 224))
    
    # Convert to NumPy array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    
    # Create a batch axis: shape becomes (1, 224, 224, 3)
    img_array = tf.expand_dims(img_array, 0) 
    
    # Get predictions
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    class_name = CLASS_NAMES[np.argmax(score)]
    confidence = float(np.max(score)) * 100.0
    
    return class_name, confidence

# --- Random Forest Inference (Price Prediction) ---

def load_rf_model():
    """Loads the trained scikit-learn pipeline."""
    if not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError(f"RF model not found at {RF_MODEL_PATH}. Did you run train_rf.py?")
    
    print("Loading RF model...")
    pipeline = joblib.load(RF_MODEL_PATH)
    return pipeline

def predict_price(state, waste_type, pipeline):
    """Predicts market price per kg based on state and waste type."""
    # Create a DataFrame that matches the format the pipeline expects
    input_df = pd.DataFrame([{
        'State': state,
        'Agricultural Waste Type': waste_type
    }])
    
    # Predict
    try:
        predicted_price = pipeline.predict(input_df)[0]
        return predicted_price
    except Exception as e:
        print(f"Price prediction failed for {waste_type} in {state}: {e}")
        return None
