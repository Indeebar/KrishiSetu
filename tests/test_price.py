import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def test_price_endpoint_success():
    """Test the /api/price endpoint with a valid state and waste type."""
    response = client.get(
        "/api/price?state=Maharashtra&waste_type=Rice_straw"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "state" in data
    assert "waste_type" in data
    assert "price_per_kg" in data
    assert data["state"] == "Maharashtra"
    assert data["waste_type"] == "Rice_straw"
    assert type(data["price_per_kg"]) == float

def test_shap_endpoint_success():
    """Test the /api/shap endpoint for feature importance."""
    response = client.get(
        "/api/shap?state=Maharashtra&waste_type=Rice_straw"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "base_value" in data
    assert "shap_contributions" in data
    assert type(data["shap_contributions"]) == list
