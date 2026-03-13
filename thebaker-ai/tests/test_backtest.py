import pytest
import pandas as pd
import numpy as np
from evaluate_backtest import calculate_metrics, run_rolling_backtest, aggregate_backtest_report

@pytest.fixture
def dummy_backtest_data():
    dates = pd.date_range("2023-01-01", "2023-03-31") # 90 days total 
    products = ['C']
    
    rows = []
    for d in dates:
        for p in products:
            sold = 100 + int(np.random.normal(0, 10))
            waste = 5 + int(np.random.normal(0, 2))
            
            rows.append({
                'date': d, 
                'product': p,
                'sold_qty': max(0, sold),
                'waste_qty': max(0, waste),
                'made_qty': max(0, sold + waste),
                'weather': 'Sunny',
                'temp_avg': 20.0,
                'sold_lag_1': 100,
                'waste_lag_1': 5,
                'dow_sin': 0.5,
                'dow_cos': 0.5,
                'month_sin': 0.5,
                'month_cos': 0.5
            })
            
    df = pd.DataFrame(rows)
    df['product'] = df['product'].astype('category')
    df['weather'] = df['weather'].astype('category')
    return df

def test_calculate_metrics():
    y_true = np.array([10, 20])
    y_pred = np.array([5, 25])
    metrics = calculate_metrics(y_true, y_pred)
    assert np.isclose(metrics['mae'], 5.0)
    assert np.isclose(metrics['wape'], 10/30)

def test_run_rolling_backtest_structure(dummy_backtest_data):
    # Testing 2 folds of 7 days
    results_df = run_rolling_backtest(
        dummy_backtest_data, 
        target_col='waste_qty', 
        n_splits=2, 
        window_size_days=7
    )
    
    # Needs 2 rows for 2 folds
    assert len(results_df) == 2
    
    # Expected analytical columns exist
    expected_cols = [
        'fold', 'target', 'val_start', 'val_end', 
        'xgb_mae', 'xgb_wape', 'lstm_wape', 'ensemble_wape'
    ]
    for c in expected_cols:
        assert c in results_df.columns
        
    # Validation dates shouldn't overlap
    assert results_df.iloc[0]['val_start'] > results_df.iloc[1]['val_end']

def test_aggregate_backtest_report():
    sold_df = pd.DataFrame({
        'xgb_wape': [0.1, 0.15],
        'lstm_wape': [0.2, 0.25],
        'ensemble_wape': [0.12, 0.18]
    })
    
    waste_df = pd.DataFrame({
        'xgb_wape': [0.3, 0.35],
        'lstm_wape': [0.1, 0.15],
        'ensemble_wape': [0.2, 0.25]
    })
    
    report = aggregate_backtest_report(sold_df, waste_df)
    
    # Sold QTY: XGB (0.125) beat Ensemble (0.15) beat LSTM (0.225) => Weight 100% XGB (default)
    assert report['recommended_weights']['sold_qty']['xgb'] == 1.0
    assert report['recommended_weights']['sold_qty']['lstm'] == 0.0
    
    # Waste QTY: LSTM (0.125) beat Ensemble (0.225) beat XGB (0.325) => Weight 90% LSTM
    assert report['recommended_weights']['waste_qty']['xgb'] == 0.1
    assert report['recommended_weights']['waste_qty']['lstm'] == 0.9
