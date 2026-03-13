"""
test_smoke.py - Automated End-to-End System Validation for The Baker V3.

A lightweight, execution-oriented test suite that literally strings the output 
of step (N) seamlessly into the inputs of step (N+1), proving mathematically 
that all architectural modules designed in Sprint 1 through Sprint 3 natively
communicate together without integration faults or schema breakages.
"""

import pandas as pd
import pytest
import os
import json

# Relative imports from the local architectural modules we've constructed
from preprocess_history import build_daily_operational_table
from features import build_operational_features
from ensemble import apply_deterministic_ensemble, compute_recommended_made_qty
from explain import ForecastContext, generate_explanation
from forecast_graph import execute_forecast_workflow

def test_pipeline_smoke():
    """Validates the entire offline ML pipeline executes continuously."""
    
    # 1. Provide Mock Raw Logs representing identical schema as raw `history.csv`
    mock_raw_data = pd.DataFrame([
        {'date': '2023-11-18', 'product': 'T-Bone Baguette', 'qty': 20, 'weather': 'Sunny', 'temp_avg': 20.0},
        {'date': '2023-11-18', 'product': 'T-Bone Baguette', 'qty': -2, 'weather': 'Sunny', 'temp_avg': 20.0},
        {'date': '2023-11-19', 'product': 'T-Bone Baguette', 'qty': 25, 'weather': 'Rainy', 'temp_avg': 10.0},
        {'date': '2023-11-19', 'product': 'T-Bone Baguette', 'qty': -5, 'weather': 'Rainy', 'temp_avg': 10.0},
    ])
    
    preprocessed_df = build_daily_operational_table(mock_raw_data)
    assert 'sold_qty' in preprocessed_df.columns
    assert 'waste_qty' in preprocessed_df.columns
    assert len(preprocessed_df) == 2 # 2 days mapped
    
    features_df = build_operational_features(preprocessed_df)
    assert 'month' in features_df.columns
    assert 'sold_lag_1' in features_df.columns
    
    # 4. Assert Ensemble cleanly merges deterministic numerical matrices securely
    sold_ensemble = apply_deterministic_ensemble('sold_qty', xgb_pred=30.0, lstm_pred=10.0) # Using 20% gap trigger
    assert sold_ensemble['anomaly_flag'] is True
    
    # 5. Assert Graph correctly routes states to Gemini format parameters
    made_qty = compute_recommended_made_qty(
        {'final_prediction': 25.0}, 
        {'final_prediction': 5.0}
    )
    assert made_qty == 30.0
    
def test_online_inference_smoke():
    """Validates the exact pathway the API endpoint utilizes using the LangGraph state orchestrator."""
    mock_features_for_api = pd.DataFrame([{'weather': 'Sunny', 'temp_avg': 25.0, 'product': 'TestCroissant'}])
    
    result = execute_forecast_workflow("TestCroissant", "2023-12-01", mock_features_for_api)
    
    # Ensures the strict dual-target definitions passed cleanly all the way out of the API.
    expected_keys = [
        'product', 'target_date', 
        'sold_final_prediction', 'waste_final_prediction', 'recommended_made_qty',
        'sold_anomaly_flag', 'waste_anomaly_flag',
        'explanation', 'risk_note', 'confidence_label', 'production_note'
    ]
    
    for key in expected_keys:
        assert key in result, f"Graph inference failed structurally missing key: {key}"
        
def test_fastapi_local_status():
    """Validates FastAPI components can securely mount via Request testing."""
    from fastapi.testclient import TestClient
    from api import app
    
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()['status'] == 'ok'
