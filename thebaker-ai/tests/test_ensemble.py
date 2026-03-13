import pytest
from ensemble import apply_deterministic_ensemble, compute_recommended_made_qty, calculate_gap_ratio
import numpy as np

def test_calculate_gap_ratio():
    assert calculate_gap_ratio(100.0, 100.0) == 0.0
    assert calculate_gap_ratio(100.0, 90.0) == 0.1  # 10 / max(100, 90)
    assert calculate_gap_ratio(0.0, 0.0) == 0.0

def test_missing_model_fallbacks():
    # Only XGB works
    res1 = apply_deterministic_ensemble('sold_qty', xgb_pred=50.0, lstm_pred=None)
    assert res1['final_prediction'] == 50.0
    assert res1['selected_strategy'] == 'fallback_primary_xgb'
    
    # Only LSTM works
    res2 = apply_deterministic_ensemble('sold_qty', xgb_pred=None, lstm_pred=60.0)
    assert res2['final_prediction'] == 60.0
    assert res2['selected_strategy'] == 'fallback_challenger_lstm'
    
    # Negative clamp enforcement natively built in
    res3 = apply_deterministic_ensemble('sold_qty', xgb_pred=-10.0, lstm_pred=None)
    assert res3['final_prediction'] == 0.0

def test_high_agreement_optimal_weights():
    # Gap = 0.0 => < 5%. 
    # 'sold_qty' DEFAULT expects 100% XGB.
    res_sold = apply_deterministic_ensemble('sold_qty', xgb_pred=100.0, lstm_pred=100.0)
    assert res_sold['selected_strategy'] == 'target_optimal_average'
    assert res_sold['final_prediction'] == 100.0
    
    # 'waste_qty' DEFAULT expects 10% XGB / 90% LSTM. 
    # Gap < 5% (100 vs 98 -> gap is 2 / 100 = 0.02)
    res_waste = apply_deterministic_ensemble('waste_qty', xgb_pred=100.0, lstm_pred=98.0)
    assert res_waste['selected_strategy'] == 'target_optimal_average'
    # 100 * 0.1 + 98 * 0.9 = 10 + 88.2 = 98.2
    assert np.isclose(res_waste['final_prediction'], 98.2)

def test_moderate_disagreement_xgb_biased():
    # Gap is exactly 10% meaning 5% < Gap <= 20%
    # xgb = 100, lstm = 90
    res = apply_deterministic_ensemble('sold_qty', xgb_pred=100.0, lstm_pred=90.0)
    assert res['selected_strategy'] == 'xgb_biased_blend'
    assert res['gap_ratio'] == 0.1
    # Bias expects 80% / 20%
    # 100 * 0.8 + 90 * 0.2 = 80 + 18 = 98.0
    assert res['final_prediction'] == 98.0

def test_vast_disagreement_anomaly():
    # Gap is 50%. (100 vs 50)
    res = apply_deterministic_ensemble('waste_qty', xgb_pred=100.0, lstm_pred=50.0)
    
    assert res['selected_strategy'] == 'anomaly_fallback_primary'
    assert res['gap_ratio'] == 0.5
    assert res['anomaly_flag'] is True
    # Falls completely back to primary
    assert res['final_prediction'] == 100.0

def test_compute_recommended_made_qty():
    sold_ens = {'final_prediction': 150.0}
    waste_ens = {'final_prediction': 15.2}
    
    made = compute_recommended_made_qty(sold_ens, waste_ens)
    assert made == 165.2
