"""
ensemble.py - Deterministic prediction reconciler for The Baker V3.

Responsible for safely combining the physical outputs of the XGBoost primary
model and the LSTM challenger. Applies strict, interpretable rules determining
final prediction outputs, generating anomaly flags if models diverge vastly.

Also supplies high-level operational helpers, like combining the final ensemble
sold_qty and waste_qty to derive a recommended_made_qty.
"""

from typing import Dict, Any, Optional

# Architecturally derived recommended weights from evaluate_backtest.py evidence:
# XGB proved dominant for general scalar tabulars (sales).
# LSTM sequences proved stronger for noise patterns like waste.
DEFAULT_WEIGHTS = {
    'sold_qty': {'xgb': 1.0, 'lstm': 0.0},
    'waste_qty': {'xgb': 0.1, 'lstm': 0.9}
}

def calculate_gap_ratio(pred1: float, pred2: float) -> float:
    """Calculates relative disparity between two predictions safely."""
    if pred1 == 0.0 and pred2 == 0.0:
        return 0.0
    
    # Normalizing against the maximum structural expectation to prevent tiny-number division blowups
    denominator = max(abs(pred1), abs(pred2))
    return abs(pred1 - pred2) / denominator

def apply_deterministic_ensemble(
    target_col: str,
    xgb_pred: Optional[float],
    lstm_pred: Optional[float],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Core reconciler. 
    1. Reconciles missing execution boundaries.
    2. Calculates model disagreement ratio.
    3. Applies deterministic mapping to generate a final production quantity.
    """
    
    # 0. Formatting - Operational quantities must not be negative mathematically
    xgb_pred = max(0.0, float(xgb_pred)) if xgb_pred is not None else None
    lstm_pred = max(0.0, float(lstm_pred)) if lstm_pred is not None else None
    
    if weights is None:
        weights = DEFAULT_WEIGHTS.get(target_col, {'xgb': 0.5, 'lstm': 0.5})

    result = {
        'target_col': target_col,
        'xgb_pred': xgb_pred,
        'lstm_pred': lstm_pred,
        'final_prediction': 0.0,
        'gap_ratio': 0.0,
        'anomaly_flag': False,
        'selected_strategy': '',
        'fallback_reason': None
    }

    # 1. Pipeline Failure Fallbacks
    if xgb_pred is None and lstm_pred is None:
        result['selected_strategy'] = 'fallback_zero'
        result['fallback_reason'] = 'Both primary and challenger missing'
        return result
        
    if lstm_pred is None:
        result['final_prediction'] = xgb_pred
        result['selected_strategy'] = 'fallback_primary_xgb'
        result['fallback_reason'] = 'Challenger (LSTM) missing or error'
        return result
        
    if xgb_pred is None:
        result['final_prediction'] = lstm_pred
        result['selected_strategy'] = 'fallback_challenger_lstm'
        result['fallback_reason'] = 'Primary (XGB) missing or error'
        return result

    # 2. Logic Paths and Gap Analysis
    gap = calculate_gap_ratio(xgb_pred, lstm_pred)
    result['gap_ratio'] = gap

    if gap <= 0.05:
        # A. High Agreement: Proceed with Evidence-backed target optimal weights
        result['final_prediction'] = (xgb_pred * weights['xgb']) + (lstm_pred * weights['lstm'])
        result['selected_strategy'] = 'target_optimal_average'
        
    elif gap <= 0.20:
        # B. Moderate Disagreement: Enforce mathematical safety anchoring to the Tabular structural base
        result['final_prediction'] = (xgb_pred * 0.8) + (lstm_pred * 0.2)
        result['selected_strategy'] = 'xgb_biased_blend'
        
    else:
        # C. Vast Disagreement: Something structurally changed. Flag anomaly, rely exclusively on Tabular base.
        result['anomaly_flag'] = True
        result['final_prediction'] = xgb_pred 
        result['selected_strategy'] = 'anomaly_fallback_primary'
        result['fallback_reason'] = 'Model gap > 20%, falling back explicitly to primary XGBoost'
        
    # Final sanity floor
    result['final_prediction'] = max(0.0, float(result['final_prediction']))
    
    return result

def compute_recommended_made_qty(
    sold_ensemble_result: Dict[str, Any],
    waste_ensemble_result: Dict[str, Any]
) -> float:
    """
    Operation helper combining independent ensembled trajectories into the 
    single structural bakery metric: How many do we need to bake?
    """
    sold_final = sold_ensemble_result.get('final_prediction', 0.0)
    waste_final = waste_ensemble_result.get('final_prediction', 0.0)
    
    # Made = Expected Demand + Expected Discards allowance
    return max(0.0, sold_final + waste_final)
