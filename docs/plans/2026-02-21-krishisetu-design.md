# KrishiSetu ML System Design

## 1. System Architecture
The KrishiSetu application will be a Streamlit-based web application that serves two primary machine learning models. The system allows farmers to upload images of their agricultural waste to identify the type of waste, and input their location (State) to get an estimated market price for that waste.
*   **Frontend**: Streamlit
*   **Image Classification**: Custom CNN (TensorFlow/Keras)
*   **Price Prediction**: Random Forest Regressor (Scikit-Learn)

## 2. Image Classification Model (Deep Learning)
*   **Objective**: Classify images of agricultural waste into one of 16 categories.
*   **Dataset**: `Agri_Waste_Images` directory containing 16 subdirectories.
*   **Model**: A simple, custom-built Convolutional Neural Network (CNN) with a few convolutional layers, max pooling, and fully connected layers. This demonstrates foundational deep learning knowledge without overly complex architectures.
*   **Framework**: TensorFlow/Keras.
*   **Deployment**: The trained model (`.keras` or `.h5`) will be loaded into the Streamlit app for inference.

## 3. Price Estimation Model (Traditional ML)
*   **Objective**: Predict the market price of agricultural waste.
*   **Dataset**: `Bio_WASTE_Cleaned.csv`.
*   **Features**: `State`, `Agricultural Waste Type`, `Demand Level`. Quantity will be fixed to a default (e.g., 1000 kg) or dropped if not highly correlated with price per kg.
*   **Target**: `Market Price /kg`.
*   **Model**: Random Forest Regressor.
*   **Preprocessing**: Categorical encoding (One-Hot or Label Encoding) for textual data.
*   **Framework**: Scikit-Learn.
*   **Deployment**: The trained model and preprocessors will be saved as a `.pkl` file via `joblib` or `pickle` to be loaded by Streamlit.

## 4. Web Application (Streamlit)
*   **User Flow**:
    1. User opens the app.
    2. User selects their `State` from a dropdown (to simulate location input).
    3. User uploads an image of the agricultural waste.
    4. Upon clicking "Analyze":
        - The image is passed through the Custom CNN model to classify the waste type.
        - The predicted waste type, along with the user's `State`, is passed to the Random Forest model to predict the market price.
        - The UI displays the classified waste type and its predicted market price per kg.
*   **Integration**: The app will load both models into memory, utilizing Streamlit caching (`@st.cache_resource`) for fast inference.
