import pandas as pd
import numpy as np
import pytest
from train_xgb import (
    prepare_xgb_data,
    temporal_split,
    calculate_metrics,
    train_xgb_model,
    predict_xgb
)

@pytest.fixture
def mock_operational_features():
    """Mock dataframe mirroring operational logic with explicit leakage variables."""
    dates = pd.date_range("2023-01-01", "2023-01-10")
    df = pd.DataFrame({
        'date': dates,
        'product': pd.Categorical(['A']*5 + ['B']*5),
        
        # Operational targets/leakage points
        'sold_qty': np.arange(10, 20),
        'waste_qty': np.arange(0, 10),
        'made_qty': np.arange(10, 20) + np.arange(0, 10),
        'net_qty': np.arange(10, 20) - np.arange(0, 10),
        'gross_sales': np.arange(100, 200, 10),
        
        # Safe historical features
        'sold_lag_1': np.arange(9, 19),
        'sold_rolling_mean_7': np.arange(8, 18).astype(float),
        'weather': pd.Categorical(['Sunny']*10)
    })
    return df

def test_prepare_xgb_data_sold_qty(mock_operational_features):
    # Pass target as sold_qty
    X, y, feat_cols = prepare_xgb_data(mock_operational_features, target_col='sold_qty')
    
    # Verify the target matches
    assert list(y.values) == list(mock_operational_features['sold_qty'].values)
    
    # Date should be dropped inherently
    assert 'date' not in X.columns
    
    # Check that X strictly excludes all leakage
    assert 'sold_qty' not in X.columns
    assert 'waste_qty' not in X.columns
    assert 'made_qty' not in X.columns
    assert 'net_qty' not in X.columns
    assert 'gross_sales' not in X.columns
    
    # Categoricals and safe features exist
    assert 'product' in X.columns
    assert 'weather' in X.columns
    assert 'sold_lag_1' in X.columns
    
def test_prepare_xgb_data_waste_qty(mock_operational_features):
    # Pass target as waste_qty
    X, y, feat_cols = prepare_xgb_data(mock_operational_features, target_col='waste_qty')
    assert list(y.values) == list(mock_operational_features['waste_qty'].values)
    
    # Made/Net/Gross must be purged
    assert 'waste_qty' not in X.columns
    assert 'sold_qty' not in X.columns
    assert 'gross_sales' not in X.columns

def test_temporal_split(mock_operational_features):
    train_df, val_df = temporal_split(mock_operational_features, split_date='2023-01-08')
    assert train_df['date'].max() < pd.to_datetime('2023-01-08')
    assert val_df['date'].min() >= pd.to_datetime('2023-01-08')
    assert len(train_df) + len(val_df) == len(mock_operational_features)

def test_calculate_metrics():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([10, 18, 33])
    metrics = calculate_metrics(y_true, y_pred)
    assert np.isclose(metrics['mae'], 1.6666666)  # 5/3
    assert np.isclose(metrics['wape'], 0.0833333) # 5/60

def test_train_and_predict(mock_operational_features):
    X, y, _ = prepare_xgb_data(mock_operational_features, target_col='sold_qty')
    
    # Ultra-simple tree fast params
    model = train_xgb_model(X, y, hyperparameters={'n_estimators': 5, 'max_depth': 2})
    preds = predict_xgb(model, X)
    
    assert len(preds) == len(X)
    assert np.all(preds >= 0)
