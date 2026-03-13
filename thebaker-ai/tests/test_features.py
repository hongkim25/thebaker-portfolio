import pytest
import pandas as pd
import numpy as np
from features import build_operational_features, get_feature_columns

@pytest.fixture
def mock_operational_data() -> pd.DataFrame:
    """Creates a mock dataset simulating preprocess_history output."""
    dates = pd.date_range("2023-01-01", periods=30)
    data = {
        'date': dates,
        'product': ['Croissant'] * 30,
        'sold_qty': [100.0] * 30,
        'waste_qty': [10.0] * 30,
        'made_qty': [110.0] * 30,
        'net_qty': [90.0] * 30,
        'gross_sales': [1000.0] * 30,
        'waste_value': [100.0] * 30,
        'weather': ['Sunny'] * 30,
        'temp_avg': [20.0] * 30,
        'temp_max': [25.0] * 30,
        'temp_min': [15.0] * 30,
        'has_positive_row': [1] * 30,
        'has_negative_row': [1] * 30,
        'positive_row_count': [1] * 30,
        'negative_row_count': [1] * 30,
        'source_row_count': [2] * 30,
        'only_negative_flag': [0] * 30,
        'multi_negative_rows_flag': [0] * 30
    }
    
    # Introduce small variation on the last element to test features
    data['sold_qty'][-1] = 50.0 
    data['waste_qty'][-1] = 50.0 
    data['made_qty'][-1] = 100.0
    
    return pd.DataFrame(data)

def test_build_operational_features(mock_operational_data):
    df_feat = build_operational_features(mock_operational_data)
    
    assert len(df_feat) == 30
    
    # Verify core feature generation presence
    expected_cols = [
        'day_of_week', 'dow_sin', 'sold_lag_1', 'waste_lag_28',
        'made_rolling_mean_7', 'sold_same_dow_mean_4', 'sell_through_rate_lag_1',
        'temp_avg_diff_1', 'is_cold_shock'
    ]
    for c in expected_cols:
        assert c in df_feat.columns
    
    # Verify Sell through rate calculation 
    # (sold=100, made=110 => 0.909...)
    # (Last row: sold=50, made=100 => 0.5)
    last_row = df_feat.iloc[-1]
    assert np.isclose(last_row['sell_through_rate'], 0.5)
    
    # Lag 1 of the last row should be the previous row's STR (100/110)
    assert np.isclose(last_row['sell_through_rate_lag_1'], 100/110)

def test_target_schema_leakage(mock_operational_data):
    df_feat = build_operational_features(mock_operational_data)
    
    # Test column extraction masking out leakage
    features = get_feature_columns(df_feat, target_col='sold_qty')
    
    # It must contain historical/weather features
    assert 'sold_lag_1' in features
    assert 'weather' in features
    assert 'temp_avg' in features
    
    # It MUST NOT contain current row exact state attributes that leak
    assert 'sold_qty' not in features
    assert 'waste_qty' not in features
    assert 'made_qty' not in features
    assert 'net_qty' not in features
    assert 'gross_sales' not in features
    assert 'sell_through_rate' not in features
    
    # Nor the categorical operational flags
    assert 'has_positive_row' not in features
    assert 'only_negative_flag' not in features

def test_leakage_prevention_rolling(mock_operational_data):
    # Alter day 1 vs day 2 aggressively
    mock_operational_data.loc[0, 'sold_qty'] = 1000
    mock_operational_data.loc[1, 'sold_qty'] = 0
    
    df_feat = build_operational_features(mock_operational_data)
    
    # Day 0 (idx 0)
    # The rolling mean at Day 0 should be NaN since shift(1) leaves nothing.
    assert pd.isna(df_feat.loc[0, 'sold_rolling_mean_7'])
    
    # Day 1 (idx 1), the sold_qty is 0. 
    # But rolling_mean_7 MUST ONLY SEE Day 0 (1000).
    assert df_feat.loc[1, 'sold_rolling_mean_7'] == 1000.0
