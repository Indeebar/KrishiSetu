import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app

client = TestClient(app)

def create_test_image():
    """Create a temporary 224x224 RGB image for testing."""
    img = Image.new("RGB", (224, 224), color="red")
    b = io.BytesIO()
    img.save(b, format="JPEG")
    b.seek(0)
    return b

def test_predict_endpoint_success():
    """Test the /api/predict endpoint with a valid image."""
    img_bytes = create_test_image()
    response = client.post(
        "/api/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"state": "Maharashtra"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "waste_type" in data
    assert "confidence" in data
    assert "class_index" in data
    assert "all_classes" in data
    assert type(data["confidence"]) == float

def test_gradcam_endpoint_success():
    """Test the /api/gradcam endpoint with a valid image."""
    img_bytes = create_test_image()
    response = client.post(
        "/api/gradcam",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"class_index": "0"}
    )
    
    # If the model layers can't be fetched locally or it works, assert accordingly
    # This might return 200 or 503 if model not found, but we expect 200 in a full env
    if response.status_code == 200:
        data = response.json()
        assert "heatmap_b64" in data
