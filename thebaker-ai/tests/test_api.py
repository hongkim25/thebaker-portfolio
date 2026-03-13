import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "The Baker V3"}

def test_forecast_endpoint_success():
    payload = {
        "product": "Baguette",
        "target_date": "2023-11-20",
        "forecast_weather": "Sunny",
        "forecast_temp": 22.5
    }
    
    response = client.post("/forecast", json=payload)
    
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()
    
    # Assert structural integrity 
    assert data["product"] == "Baguette"
    assert data["target_date"] == "2023-11-20"
    
    # Assert numerical dual-targeting maps cleanly
    assert "predicted_sold_qty" in data
    assert "predicted_waste_qty" in data
    assert "recommended_made_qty" in data
    
    # Explanation fields exist
    assert "explanation" in data
    assert "production_note" in data
    assert data["confidence_label"] in ["High", "Moderate", "Low", "Moderate (Fallback Gen)"]
    
    # Anomaly fields exist
    assert isinstance(data["sold_anomaly_flag"], bool)
    assert isinstance(data["waste_anomaly_flag"], bool)
    
    # Metadata internals
    meta = data["metadata"]
    assert "sold_xgb" in meta
    assert "sold_lstm" in meta
    assert "selected_strategies" in meta

def test_forecast_endpoint_invalid_date():
    payload = {
        "product": "Baguette",
        "target_date": "invalid-date",
        "forecast_weather": "Sunny",
        "forecast_temp": 22.5
    }
    
    response = client.post("/forecast", json=payload)
    assert response.status_code == 400
    assert "Invalid isoformat string" in response.json()["detail"] or "invalid" in response.json()["detail"].lower()

def test_forecast_endpoint_missing_fields():
    # Omitting forecast_temp (required)
    payload = {
        "product": "Baguette",
        "target_date": "2023-11-20",
        "forecast_weather": "Sunny"
    }
    
    response = client.post("/forecast", json=payload)
    assert response.status_code == 422 # Pydantic validation error
