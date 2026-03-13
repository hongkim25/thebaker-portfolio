import pytest
import pandas as pd
import numpy as np
import torch
from train_lstm import (
    prepare_lstm_data,
    BakeryLSTM,
    calculate_lstm_metrics
)

@pytest.fixture
def mock_lstm_operational_data():
    """Mock dataframe mirroring operational logic."""
    dates = pd.date_range("2023-01-01", periods=60)
    
    # Ensure there are columns aligning with historic and future extraction limits
    df1 = pd.DataFrame({
        'date': dates,
        'product': ['A'] * 60,
        'sold_qty': np.arange(60),
        'waste_qty': np.arange(0, 60),
        'made_qty': np.arange(60) * 2,
        'temp_avg': np.arange(60).astype(float),
        'dow_sin': [0.5] * 60,
        'dow_cos': [0.5] * 60,
        'month_sin': [0.5] * 60,
        'month_cos': [0.5] * 60,
        'weather': ['Sunny'] * 60
    })
    
    df2 = df1.copy()
    df2['product'] = 'B'
    df2['weather'] = 'Rainy'
    df2['sold_qty'] = np.arange(60) * 3
    
    return pd.concat([df1, df2]).reset_index(drop=True)

def test_prepare_lstm_data_shapes(mock_lstm_operational_data):
    # Predict waste_qty for 14 day sequences
    dataset, s_hist, s_fut, p_map, w_map = prepare_lstm_data(
        mock_lstm_operational_data, 
        target_col='waste_qty', 
        seq_len=14
    )
    
    # 60 days per product. 60 - 14 sequences = 46. Two products = 92 sequences total.
    assert len(dataset) == 92
    
    x_hist, x_fut, x_prod, x_tw, y = dataset[0]
    
    # x_hist represents T-14 to T-1 logic
    # Contains: sold, waste, made, temp, 4 calendars => 8 continuous columns 
    assert x_hist.shape == (14, 8)
    
    # x_fut represents T known context 
    # Contains: temp, 4 calendars => 5 continuous columns
    assert x_fut.shape == (5,)
    
    # Categorical components are scalar embeddings
    assert x_prod.shape == () 
    assert x_tw.shape == ()
    
    # Verification of Target
    assert y.shape == (1,)

def test_lstm_forward_pass_multi_input(mock_lstm_operational_data):
    dataset, s_hist, s_fut, p_map, w_map = prepare_lstm_data(
        mock_lstm_operational_data, seq_len=14, target_col="sold_qty"
    )
    
    x_h, x_f, x_p, x_tw, y = dataset[0]
    
    # Build batch of 2
    x_h = torch.stack([x_h, x_h])
    x_f = torch.stack([x_f, x_f])
    x_p = torch.stack([x_p, x_p])
    x_tw = torch.stack([x_tw, x_tw])
    
    model = BakeryLSTM(
        num_hist_features=len(dataset.historic_continuous_cols), # 8
        num_fut_features=len(dataset.future_continuous_cols),   # 5
        num_products=len(p_map),
        num_weather=len(w_map)
    )
    
    preds = model(x_h, x_f, x_p, x_tw)
    
    # Batch size 2, 1 target unit
    assert preds.shape == (2, 1)

def test_lstm_metrics_nonnegative_logic():
    y_true = np.array([10, 20])
    y_pred = np.array([5, 25])
    
    # MAE = (5 + 5) / 2 = 5.0
    # WAPE = 10 / 30 = 0.33
    
    metrics = calculate_lstm_metrics(y_true, y_pred)
    assert np.isclose(metrics['mae'], 5.0)
    assert np.isclose(metrics['wape'], 10/30)
