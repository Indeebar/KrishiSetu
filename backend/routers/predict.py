"""
POST /api/predict
Accepts an uploaded image, returns predicted waste class + confidence.
Also saves the prediction to the history DB.
"""

import io
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from PIL import Image
import tensorflow as tf

from backend.db import insert_prediction

router = APIRouter()

# ── Lazy-load the CNN model once ─────────────────────────────────────────────
_cnn_model = None
_class_names = None


def get_model():
    global _cnn_model, _class_names
    if _cnn_model is None:
        import os
        from pathlib import Path
        model_path = Path(__file__).parent.parent.parent / "models" / "custom_cnn_model.keras"
        data_dir   = Path(__file__).parent.parent.parent / "Agri_Waste_Images"

        if not model_path.exists():
            raise FileNotFoundError(f"CNN model not found at {model_path}")

        _cnn_model   = tf.keras.models.load_model(str(model_path))
        _class_names = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(data_dir / d)
        ])
    return _cnn_model, _class_names


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    state: str = Form(default="Unknown"),
):
    """
    Upload an image of agricultural waste.
    Returns the predicted waste class, confidence %, and predicted market price.
    """
    try:
        model, class_names = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Read & preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # model has preprocess_input baked in

    # Inference
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    class_index = int(np.argmax(score))
    class_name  = class_names[class_index]
    confidence  = float(np.max(score)) * 100.0

    # Persist to history (price filled in by /api/price separately)
    insert_prediction(
        waste_type=class_name,
        confidence=confidence,
        state=state,
        price_per_kg=None,
        image_name=file.filename,
    )

    return {
        "waste_type":   class_name,
        "confidence":   round(confidence, 2),
        "class_index":  class_index,
        "all_classes":  class_names,
    }
