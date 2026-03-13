"""
evaluate_backtest.py - Rolling-origin evaluation framework for The Baker V3.

Responsible for orchestrating time-based validation splits to fairly compare
architectural performances across completely distinct operational targets:
- sold_qty
- waste_qty

Trains XGBoost, LSTM, and a naive Simple Average Ensemble natively within each fold.
Generates an evidence layer summary to decide weights for final ensembling.
"""

import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from train_xgb import prepare_xgb_data, train_xgb_model, predict_xgb
from train_lstm import train_lstm_model, prepare_lstm_data, BakeryLSTM
import torch

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MAE and WAPE consistently."""
    mae = np.mean(np.abs(y_true - y_pred))
    sum_actuals = np.sum(y_true)
    wape = np.sum(np.abs(y_true - y_pred)) / sum_actuals if sum_actuals > 0 else 0.0
    return {'mae': mae, 'wape': wape}

def run_rolling_backtest(df: pd.DataFrame, target_col: str, 
                         n_splits: int = 3, window_size_days: int = 14) -> pd.DataFrame:
    """
    Executes a strict time-series rolling origin cross-validation.
    Ensures Train and Val sets never leak future boundaries per fold.
    Calculates independent predictions for XGB, LSTM, and Provisional Ensemble.
    """
    df = df.copy()
    
    # Sort and collect unique chronological boundaries
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    dates = df['date'].unique()
    
    if len(dates) < window_size_days * n_splits:
        raise ValueError(f"Insufficient dates ({len(dates)}) to perform {n_splits} splits of {window_size_days} days.")
        
    results = []
    
    for i in range(n_splits):
        # Determine exact dates for this fold (Walking Backwards from newest)
        # Split 0: Newest window
        # Split 1: Previous window, etc.
        val_start_idx = len(dates) - (i + 1) * window_size_days
        val_end_idx = val_start_idx + window_size_days - 1
        
        val_start_date = dates[val_start_idx]
        val_end_date = dates[val_end_idx]
        
        # Strict temporal split BEFORE modeling steps
        train_df = df[df['date'] < val_start_date].copy()
        val_df = df[(df['date'] >= val_start_date) & (df['date'] <= val_end_date)].copy()
        
        # ----------------------------------------------------
        # 1. XGBoost Pipeline
        # ----------------------------------------------------
        X_train_xgb, y_train_xgb, _ = prepare_xgb_data(train_df, target_col)
        X_val_xgb, y_val_xgb, _ = prepare_xgb_data(val_df, target_col)
        
        xgb_model = train_xgb_model(X_train_xgb, y_train_xgb)
        xgb_preds = predict_xgb(xgb_model, X_val_xgb)
        
        # ----------------------------------------------------
        # 2. PyTorch LSTM Pipeline
        # ----------------------------------------------------
        # LSTM needs a sequence trail to evaluate the val_df block effectively, 
        # so we pass train_df sequence lengths + val_df. 
        # Crucially, prepare_lstm_data builds slices sequentially and we only eval the dates hitting val_df.
        
        # Small constraint: to evaluate val_df purely, we technically need the preceding `seq_len` days included historically
        # to construct the sequences. We simulate this by appending `seq_len` days from train to val_df internally.
        lstm_seq_len = 14
        
        # We need validation frames specifically for the target day contexts
        # We must pull trailing context from train_df
        trailing_context = train_df[train_df['date'] >= (val_start_date - pd.Timedelta(days=lstm_seq_len))]
        val_combined_for_lstm = pd.concat([trailing_context, val_df]).sort_values(['product', 'date']).reset_index(drop=True)
        
        # Train LSTM
        lstm_model, s_hist, s_fut, p_map, w_map, _ = train_lstm_model(
            train_df, val_df=None, target_col=target_col, seq_len=lstm_seq_len, epochs=10, batch_size=32
        )
        
        # Generate LSTM Predictions natively
        val_dataset, _, _, _, _ = prepare_lstm_data(val_combined_for_lstm, target_col, seq_len=lstm_seq_len,
                                                    scaler_hist=s_hist, scaler_fut=s_fut,
                                                    product_map=p_map, weather_map=w_map)
        
        lstm_preds = []
        lstm_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for x_h, x_f, x_p, x_tw, _ in torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False):
                preds = lstm_model(x_h.to(device), x_f.to(device), x_p.to(device), x_tw.to(device))
                preds = torch.clamp(preds, min=0.0) # operational floor limits
                lstm_preds.extend(preds.cpu().numpy().flatten())
                
        lstm_preds = np.array(lstm_preds)
        
        # ----------------------------------------------------
        # 3. Ensemble & Metrics Calculation
        # ----------------------------------------------------
        y_true = y_val_xgb.values # canonical truth
        
        # Edge case handling if sequence construction dropped some lengths (e.g. products missing histories)
        if len(y_true) != len(lstm_preds):
            # Simplest fallback to align lengths for backtest demonstration if sequences get truncated
            # Real enterprise deployments carefully pad/impute, but we truncate matching lengths for POC backtest.
            min_len = min(len(y_true), len(lstm_preds))
            y_true = y_true[-min_len:]
            xgb_preds = xgb_preds[-min_len:]
            lstm_preds = lstm_preds[-min_len:]
            
        ensemble_preds = (xgb_preds + lstm_preds) / 2.0
            
        xgb_metrics = calculate_metrics(y_true, xgb_preds)
        lstm_metrics = calculate_metrics(y_true, lstm_preds)
        ens_metrics = calculate_metrics(y_true, ensemble_preds)
        
        results.append({
            'fold': i,
            'target': target_col,
            'val_start': val_start_date,
            'val_end': val_end_date,
            'xgb_mae': xgb_metrics['mae'],
            'xgb_wape': xgb_metrics['wape'],
            'lstm_mae': lstm_metrics['mae'],
            'lstm_wape': lstm_metrics['wape'],
            'ensemble_mae': ens_metrics['mae'],
            'ensemble_wape': ens_metrics['wape']
        })
        
    return pd.DataFrame(results)

def aggregate_backtest_report(sold_df: pd.DataFrame, waste_df: pd.DataFrame) -> Dict[str, Any]:
    """Generates a structured portfolio evidence output representing target dominances."""
    report = {
        'sold_qty': {
            'xgb_mean_wape': sold_df['xgb_wape'].mean(),
            'lstm_mean_wape': sold_df['lstm_wape'].mean(),
            'ensemble_mean_wape': sold_df['ensemble_wape'].mean()
        },
        'waste_qty': {
            'xgb_mean_wape': waste_df['xgb_wape'].mean(),
            'lstm_mean_wape': waste_df['lstm_wape'].mean(),
            'ensemble_mean_wape': waste_df['ensemble_wape'].mean()
        },
        'recommended_weights': {}
    }
    
    # Recommendation Logic explicitly separating components
    # If XGB is vastly superior, weight it 1.0. If Ensemble wins, weight 0.5/0.5
    for target in ['sold_qty', 'waste_qty']:
        metrics = report[target]
        if metrics['ensemble_mean_wape'] < metrics['xgb_mean_wape'] and metrics['ensemble_mean_wape'] < metrics['lstm_mean_wape']:
            report['recommended_weights'][target] = {'xgb': 0.5, 'lstm': 0.5}
        elif metrics['lstm_mean_wape'] < metrics['xgb_mean_wape']:
            report['recommended_weights'][target] = {'xgb': 0.1, 'lstm': 0.9} # Strong LSTM bias
        else:
            report['recommended_weights'][target] = {'xgb': 1.0, 'lstm': 0.0} # Primary rules intact (XGB default)
            
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rolling backtest across models and targets.")
    parser.add_argument("--input", "-i", type=str, default="features_clean.csv", 
                        help="Path to precomputed features CSV")
    parser.add_argument("--splits", type=int, default=2, help="Number of rolling folds")
    parser.add_argument("--window", type=int, default=7, help="Days per validation window")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        df = pd.read_csv(args.input)
        for col in ['product', 'weather']:
            if col in df.columns:
                df[col] = df[col].astype('category')
    else:
        print(f"Warning: {args.input} not found. Generating dummy operations.")
        dates = pd.date_range("2023-01-01", "2023-04-30")
        products = ['Croissant', 'Baguette']
        
        rows = []
        for d in dates:
            for p in products:
                base_qty = 50 + (20 if d.dayofweek in [5,6] else 0)
                sold = base_qty + int(np.random.normal(0, 5))
                waste = int(np.random.beta(2,5) * 10)
                
                rows.append({
                    'date': d.strftime("%Y-%m-%d"), 
                    'product': p,
                    'sold_qty': sold,
                    'waste_qty': waste,
                    'made_qty': sold + waste,
                    'weather': 'Sunny',
                    'temp_avg': 20.0,
                    'sold_lag_1': max(0, sold + int(np.random.normal(0,2))),
                    'waste_lag_1': max(0, waste + int(np.random.normal(0,1))),
                    'sold_rolling_mean_7': float(base_qty),
                    
                    # LSTM requires these specifically
                    'dow_sin': np.sin(2 * np.pi * d.dayofweek / 7),
                    'dow_cos': np.cos(2 * np.pi * d.dayofweek / 7),
                    'month_sin': 0.5,
                    'month_cos': 0.5
                })
        df = pd.DataFrame(rows)
        for col in ['product', 'weather']:
            df[col] = df[col].astype('category')
            
    print("\n[Backtest 1/2] Evaluating 'sold_qty' target...")
    df_sold_results = run_rolling_backtest(df, target_col='sold_qty', n_splits=args.splits, window_size_days=args.window)
    print(df_sold_results[['fold', 'xgb_wape', 'lstm_wape', 'ensemble_wape']])
    
    print("\n[Backtest 2/2] Evaluating 'waste_qty' target...")
    df_waste_results = run_rolling_backtest(df, target_col='waste_qty', n_splits=args.splits, window_size_days=args.window)
    print(df_waste_results[['fold', 'xgb_wape', 'lstm_wape', 'ensemble_wape']])
    
    print("\n================== Evidence Report ==================")
    report = aggregate_backtest_report(df_sold_results, df_waste_results)
    import json
    print(json.dumps(report, indent=2))
    print("=====================================================")
