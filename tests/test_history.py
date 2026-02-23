import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.db import init_db, insert_prediction

client = TestClient(app)

def setup_module(module):
    """Setup a clean test database."""
    init_db()
    # Insert a dummy record to ensure history is not empty
    insert_prediction(
        waste_type="Apple_pomace",
        confidence=95.5,
        state="Maharashtra",
        price_per_kg=10.5,
        image_name="test.jpg"
    )

def test_history_endpoint_success():
    """Test the /api/history endpoint returns past predictions."""
    response = client.get("/api/history?limit=5")
    
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "predictions" in data
    assert type(data["predictions"]) == list
    assert data["count"] <= 5
    
    # Check the dummy record
    if data["count"] > 0:
        first_record = data["predictions"][0]
        assert "waste_type" in first_record
        assert "confidence" in first_record
        assert "price_per_kg" in first_record
        assert "timestamp" in first_record
