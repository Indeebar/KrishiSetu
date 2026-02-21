# KrishiSetu | Agri-Waste Valuator 🌾

KrishiSetu is an AI-powered agricultural waste valuation system. It uses computer vision (MobileNetV2 Transfer Learning) to automatically classify images of agricultural waste and uses a Random Forest Regressor to estimate the current market price of the waste based on the user's location.

## Features
*   **Image Classification**: Identify agricultural waste type directly from photos.
*   **Price Prediction**: Uses recent market data to give realistic valuations for the waste based on the user's state.
*   **Easy-to-use Interface**: A clean UI built with Streamlit.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd KrishiSetu
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .\.venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Repository Structure

*   `app.py`: Main Streamlit application frontend.
*   `src/models/train_cnn.py`: Script to fine-tune the MobileNetV2 image classification model.
*   `src/models/train_rf.py`: Script to train the Random Forest price prediction model.
*   `src/utils/inference.py`: Core logic combining predicting classes and formatting data for UI consumption.
*   `models/`: Directory where trained `.keras` and `.pkl` models are saved.
*   `Agri_Waste_Images/`: Image dataset directory containing subfolders for each class.
*   `Bio_WASTE_Cleaned.csv`: Tabular dataset for training the pricing model.

## Model Details
*   **Image Classification**: MobileNetV2 pre-trained on ImageNet, fine-tuned on custom Agri-Waste data.
*   **Price Evaluation**: RandomForestRegressor trained on categorical data mapping waste types and geographical states to output market values.
