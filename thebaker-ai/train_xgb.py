"""
train_xgb.py - Primary production model training using XGBoost.

Responsible for training an XGBoost model for tabular forecasting
of next-day product demand using rolling-origin time-based validation.

Design Constraints:
- NO RANDOM SPLITS: Time-series forecasting uses time-based splitting.
- NO TARGET LEAKAGE: Operational quantities like 'made_qty' and 'gross_sales'
  are masked out dynamically by features.get_feature_columns().
- MULTI-TARGET SUPPORT: Separate models trained individually for:
  - sold_qty
  - waste_qty
"""

import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from typing import Tuple, Dict, Any, List, Optional
from features import get_feature_columns

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MAE and WAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    
    sum_actuals = np.sum(y_true)
    if sum_actuals == 0:
        wape = 0.0
    else:
        wape = np.sum(np.abs(y_true - y_pred)) / sum_actuals
        
    return {
        'mae': mae,
        'wape': wape
    }

def temporal_split(df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a strict time-based split at the given date.
    Training data is strictly < split_date.
    Validation data is >= split_date.
    """
    split_date_dt = pd.to_datetime(split_date)
    train_df = df[df['date'] < split_date_dt].copy()
    val_df = df[df['date'] >= split_date_dt].copy()
    return train_df, val_df

def prepare_xgb_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Separates the operational dataframe into safe features (X) and target (y).
    Automatically calls `features.py:get_feature_columns()` to strip 
    leakage-prone same-row metrics entirely out of the input set.
    """
    feature_cols = get_feature_columns(df, target_col)
    
    # Ensure subset logic works safely even if all target_cols aren't technically present
    # in some tests
    existing_feat_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[existing_feat_cols]
    
    if target_col in df.columns:
        y = df[target_col]
    else:
        y = pd.Series(dtype=np.float32)
        
    return X, y, existing_feat_cols

def train_xgb_model(X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                    hyperparameters: Optional[Dict[str, Any]] = None) -> xgb.XGBRegressor:
    """
    Trains an XGBoost Regressor.
    If validation data is provided, utilizes early stopping.
    """
    default_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'enable_categorical': True,
        'tree_method': 'hist',
        'random_state': 42
    }
    
    if hyperparameters:
        default_params.update(hyperparameters)
        
    model = xgb.XGBRegressor(**default_params)
    
    if X_val is not None and len(X_val) > 0 and y_val is not None and len(y_val) > 0:
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train, verbose=False)
        
    return model

def predict_xgb(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """Generates predictions clamping negative values to 0 since operational qty cannot be negative."""
    preds = model.predict(X)
    return np.clip(preds, a_min=0, a_max=None)

def save_xgb_model(model: xgb.XGBRegressor, filepath: str) -> None:
    """Saves the XGBoost model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save_model(filepath)

def load_xgb_model(filepath: str) -> xgb.XGBRegressor:
    """Loads an XGBoost model from disk."""
    model = xgb.XGBRegressor()
    model.load_model(filepath)
    return model

# ---------------------------------------------------------------------------
# Simple rolling origin example for evaluate_backtest usage

def rolling_origin_cv(df: pd.DataFrame, target_col: str, n_splits: int, window_size: int) -> pd.DataFrame:
    """
    Rolling-origin time-based validation for XGBoost.
    Instead of random splits, we move an evaluation window forward in time.
    """
    df = df.copy()
    dates = df['date'].sort_values().unique()
    
    if len(dates) < window_size * n_splits:
        raise ValueError(f"Not enough dates ({len(dates)}) to support requested splits ({n_splits}) and window size ({window_size}).")
        
    results = []
    
    for i in range(n_splits):
        val_start_idx = len(dates) - (n_splits - i) * window_size
        val_start_date = dates[val_start_idx]
        val_end_date = dates[val_start_idx + window_size - 1] if val_start_idx + window_size - 1 < len(dates) else dates[-1]
        
        train_df = df[df['date'] < val_start_date]
        val_df = df[(df['date'] >= val_start_date) & (df['date'] <= val_end_date)]
        
        X_train, y_train, _ = prepare_xgb_data(train_df, target_col=target_col)
        X_val, y_val, _ = prepare_xgb_data(val_df, target_col=target_col)
        
        model = train_xgb_model(X_train, y_train)
        preds = predict_xgb(model, X_val)
        
        metrics = calculate_metrics(y_val.values, preds)
        
        results.append({
            'split': i,
            'target': target_col,
            'train_end': train_df['date'].max(),
            'val_start': val_start_date,
            'val_end': val_end_date,
            'mae': metrics['mae'],
            'wape': metrics['wape']
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model on preprocessed table.")
    parser.add_argument("--target", "-t", type=str, default="sold_qty", 
                        choices=["sold_qty", "waste_qty"], 
                        help="Target operational col to forecast (default: sold_qty)")
    parser.add_argument("--input", "-i", type=str, default="features_clean.csv", 
                        help="Path to precomputed features CSV")
    
    args = parser.parse_args()
    
    print(f"[{args.target}] Initializing model training...")
    
    # Try to load local file, else generate fake dummy data
    if os.path.exists(args.input):
        print(f"Loading {args.input}")
        df = pd.read_csv(args.input)
        df['date'] = pd.to_datetime(df['date'])
        for col in ['product', 'weather']:
            if col in df.columns:
                df[col] = df[col].astype('category')
    else:
        print(f"File {args.input} not found. Generating minimal dummy features to test pipeline.")
        # Need enough dates to run the test
        dates = pd.date_range("2023-01-01", "2023-04-01") 
        products = ['Croissant', 'Baguette']
        
        rows = []
        for d in dates:
            for p in products:
                base_qty = 50 + (20 if d.dayofweek in [5,6] else 0)
                sold = base_qty + int(np.random.normal(0, 5))
                waste = int(np.random.beta(2,5) * 10)
                
                rows.append({
                    'date': d, # datetime
                    'product': p,
                    'sold_qty': sold,
                    'waste_qty': waste,
                    'made_qty': sold + waste,
                    'weather': 'Sunny',
                    'temp_avg': 20.0,
                    'sold_lag_1': max(0, sold + int(np.random.normal(0,2))),
                    'waste_lag_1': max(0, waste + int(np.random.normal(0,1))),
                    'sold_rolling_mean_7': float(base_qty)
                })
        df = pd.DataFrame(rows)
        df['weather'] = df['weather'].astype('category')
        df['product'] = df['product'].astype('category')
        
    # Standard splitting (e.g., hold out last 21 days for eval)
    max_date = df['date'].max()
    split_date = max_date - pd.Timedelta(days=21)
    
    print(f"Splitting data temporally at {split_date}...")
    train_df, val_df = temporal_split(df, split_date=split_date)
    
    # Extract features mapping safely
    X_train, y_train, feat_cols = prepare_xgb_data(train_df, target_col=args.target)
    X_val, y_val, _ = prepare_xgb_data(val_df, target_col=args.target)
    
    print(f"Training XGB model targeting {args.target} using {len(feat_cols)} tracking features...")
    # Hide verbose early stopping in main script output usually
    model = train_xgb_model(X_train, y_train, X_val, y_val)
    
    preds = predict_xgb(model, X_val)
    metrics = calculate_metrics(y_val.values, preds)
    
    print(f"\n[{args.target}] Validation Results:")
    print(f"MAE:  {metrics['mae']:.2f}")
    print(f"WAPE: {metrics['wape']:.2%}")
    
    save_path = f"models/xgb_{args.target}.json"
    print(f"Saving to {save_path}...")
    save_xgb_model(model, save_path)
    print("Done.")
