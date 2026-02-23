"""
POST /api/gradcam
Accepts an uploaded image + class_index, returns a Grad-CAM heatmap overlay
as a base64-encoded PNG string.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which pixel
regions of the image the CNN focused on when making its classification decision.
"""

import io
import base64
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from PIL import Image
import tensorflow as tf
import cv2

router = APIRouter()


def _get_gradcam_heatmap(model, img_array: np.ndarray, class_index: int) -> np.ndarray:
    """
    Computes a Grad-CAM heatmap for the given image and class index.
    Works by finding the nested MobileNetV2 base model and its last conv layer.
    """
    # 1. Recursive search to find the last Conv2D layer ANYWHERE in the model hierarchy
    def find_last_conv_layer(keras_model):
        last_conv = None
        for layer in reversed(keras_model.layers):
            if isinstance(layer, tf.keras.Model):
                res = find_last_conv_layer(layer)
                if res: return res
            elif isinstance(layer, tf.keras.layers.Conv2D):
                return layer
        return last_conv

    last_conv_layer = find_last_conv_layer(model)
    if last_conv_layer is None:
        raise ValueError("Could not find any Conv2D layer in the entire model.")

    # 2. To avoid Keras internal functional graph slicing bugs with nested models,
    # we will extract the exact name of the layer in the internal execution graph.
    # In MobileNetV2, the last conv is 'Conv_1' (which is the actual layer, but inside 'mobilenetv2_1.00_224')
    
    # We create a single continuous sub-model pointing directly to the internal layer's output tensor
    base_model = model.get_layer("mobilenetv2_1.00_224")
    internal_conv_output = None
    
    for l in reversed(base_model.layers):
        if isinstance(l, tf.keras.layers.Conv2D):
            internal_conv_output = l.output
            break
            
    if internal_conv_output is None:
        raise ValueError("Failed to find Conv2D output inside MobileNetV2")

    # This creates a model taking raw image input and producing the base model's internal conv output
    # But wait, our custom model expects raw images, augmenting, THEN passing to base_model.
    # The safest Grad-CAM for this specific architecture: calculate gradients on the base_model standalone.
    
    # Keras 3 safe Grad-CAM for MobileNetV2 feature extractor:
    # 1) Preprocess input manually (skipping custom augmentation layer to avoid graph trace issues)
    img_array_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(tf.identity(img_array))
    
    conv_model = tf.keras.Model(base_model.inputs, internal_conv_output)
    
    # We must redefine the classifier head explicitly using the existing trained weights
    # to avoid Functional.call() graph errors with the nested base_model.
    # Fortunately, the head is simple: GlobalAveragePooling2D -> Dropout -> Dense
    pool_layer = model.get_layer("global_average_pooling2d")
    dense_layer = model.get_layer("dense")
    
    with tf.GradientTape() as tape:
        # Get conv map
        conv_outputs = conv_model(img_array_preprocessed)
        tape.watch(conv_outputs)
        
        # Manually pass through the head
        x = pool_layer(conv_outputs)
        # Dropout is skipped in inference Mode
        predictions = dense_layer(x)
        
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap


def _overlay_heatmap(original_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> str:
    """
    Resizes heatmap to image size, applies jet colormap, blends with original,
    and returns a base64-encoded PNG string.
    """
    img_np   = np.array(original_pil)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (img_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    result_pil = Image.fromarray(overlay)

    buffer = io.BytesIO()
    result_pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


@router.post("/gradcam")
async def gradcam(
    file: UploadFile = File(...),
    class_index: int = Form(...),
):
    """
    Generate a Grad-CAM heatmap overlay for the uploaded image.
    Returns: { heatmap_b64: "<base64 PNG string>" }
    """
    from backend.routers.predict import get_model
    try:
        model, _ = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    contents = await file.read()
    original = Image.open(io.BytesIO(contents)).convert("RGB")
    resized  = original.resize((224, 224))

    img_array = tf.keras.preprocessing.image.img_to_array(resized)
    img_array = tf.expand_dims(img_array, 0)

    try:
        heatmap = _get_gradcam_heatmap(model, img_array, class_index)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {e}")

    heatmap_b64 = _overlay_heatmap(resized, heatmap)
    return {"heatmap_b64": heatmap_b64}
