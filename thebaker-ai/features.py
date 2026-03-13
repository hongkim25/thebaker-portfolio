"""
features.py - Feature engineering module for The Baker V3.

Responsible for transforming the preprocessed daily operational table
into model-ready feature sets. Calculates calendar properties, operational lags, 
rolling statistics, and ratio features strictly using historical non-leaking slices.

This is the canonical feature engineering layer for downstream models.
"""

import pandas as pd
import numpy as np
from typing import List

REQUIRED_COLUMNS = {
    'date', 'product', 'sold_qty', 'waste_qty', 'made_qty', 'net_qty',
    'gross_sales', 'waste_value', 'weather', 'temp_avg', 'temp_max', 'temp_min',
    'has_positive_row', 'has_negative_row', 'positive_row_count', 'negative_row_count',
    'source_row_count', 'only_negative_flag', 'multi_negative_rows_flag'
}

def load_preprocessed_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates schema and ensures date sorting."""
    df = df.copy()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns from preprocessed table: {missing}")
        
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['product', 'date']).reset_index(drop=True)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds standard time properties."""
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    # isocalendar().week returns UInt32 by default in pandas, casting to int
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    
    # Cyclical encodings
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    return df

def add_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """Adds straightforward historical lags for the 3 core qty columns."""
    for lag in lags:
        df[f'sold_lag_{lag}'] = df.groupby('product')['sold_qty'].shift(lag)
        df[f'waste_lag_{lag}'] = df.groupby('product')['waste_qty'].shift(lag)
        df[f'made_lag_{lag}'] = df.groupby('product')['made_qty'].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Calculates rolling statistics safely using shift(1)."""
    for w in windows:
        # Crucial: Shift by 1 first to prevent current-row target leakage
        shifted_sold = df.groupby('product')['sold_qty'].shift(1)
        shifted_waste = df.groupby('product')['waste_qty'].shift(1)
        shifted_made = df.groupby('product')['made_qty'].shift(1)
        
        df[f'sold_rolling_mean_{w}'] = shifted_sold.groupby(df['product']).rolling(window=w, min_periods=1).mean().reset_index(0, drop=True)
        df[f'waste_rolling_mean_{w}'] = shifted_waste.groupby(df['product']).rolling(window=w, min_periods=1).mean().reset_index(0, drop=True)
        df[f'made_rolling_mean_{w}'] = shifted_made.groupby(df['product']).rolling(window=w, min_periods=1).mean().reset_index(0, drop=True)
        
        if w == 7: # Standard requirement: Std Dev for past 7 days
            df[f'sold_rolling_std_{w}'] = shifted_sold.groupby(df['product']).rolling(window=w, min_periods=2).std().reset_index(0, drop=True)
            df[f'waste_rolling_std_{w}'] = shifted_waste.groupby(df['product']).rolling(window=w, min_periods=2).std().reset_index(0, drop=True)
            df[f'made_rolling_std_{w}'] = shifted_made.groupby(df['product']).rolling(window=w, min_periods=2).std().reset_index(0, drop=True)
            
    return df

def add_same_weekday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds same-weekday averages (e.g. past 4 mondays) 
    by grouping on both [product, day_of_week].
    """
    # Create shifted arrays to group without leaking
    shifted_sold = df.groupby('product')['sold_qty'].shift(7) # To guarantee at least 1 week ago
    shifted_waste = df.groupby('product')['waste_qty'].shift(7)
    shifted_made_ = df.groupby('product')['made_qty'].shift(7)
    
    # We can calculate moving averages of the same day-of-week by taking every 7th element of the raw sequences,
    # or by grouping on (product, dow) and rolling. The latter is structurally cleaner if the dataset is dense.
    # Note: If missing dates exist, rolling by 4 on DOW actually means past 4 observed DOWs.
    # Preprocess_history output isn't explicitly missing-date filled yet, so we'll explicitly shift by multiples of 7.
    
    # Simpler and safer for dense temporal datasets: explicitly stack exact lags
    for num_weeks in [4, 8]:
        lags = [w * 7 for w in range(1, num_weeks + 1)]
        sold_stack = np.column_stack([df.groupby('product')['sold_qty'].shift(l) for l in lags])
        waste_stack = np.column_stack([df.groupby('product')['waste_qty'].shift(l) for l in lags])
        made_stack = np.column_stack([df.groupby('product')['made_qty'].shift(l) for l in lags])
        
        # np.nanmean ignores missing historical lags safely
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df[f'sold_same_dow_mean_{num_weeks}'] = np.nanmean(sold_stack, axis=1)
            df[f'waste_same_dow_mean_{num_weeks}'] = np.nanmean(waste_stack, axis=1)
            df[f'made_same_dow_mean_{num_weeks}'] = np.nanmean(made_stack, axis=1)
            
            if num_weeks == 8:
                df[f'sold_same_dow_std_8'] = np.nanstd(sold_stack, axis=1)
                df[f'waste_same_dow_std_8'] = np.nanstd(waste_stack, axis=1)
                df[f'made_same_dow_std_8'] = np.nanstd(made_stack, axis=1)
                
    return df

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates operational ratios with zero-safe division."""
    
    def safe_div(num, denom):
        return np.where(denom == 0, 0.0, num / denom)
        
    df['sell_through_rate'] = safe_div(df['sold_qty'], df['made_qty'])
    df['waste_rate'] = safe_div(df['waste_qty'], df['made_qty'])
    
    # Pure lags of ratios
    for lag in [1, 7]:
        df[f'sell_through_rate_lag_{lag}'] = df.groupby('product')['sell_through_rate'].shift(lag)
        df[f'waste_rate_lag_{lag}'] = df.groupby('product')['waste_rate'].shift(lag)
        
    # Moving average of ratios (shifted)
    for w in [7, 28]:
        shifted_str = df.groupby('product')['sell_through_rate'].shift(1)
        shifted_wr = df.groupby('product')['waste_rate'].shift(1)
        df[f'sell_through_rate_rolling_mean_{w}'] = shifted_str.groupby(df['product']).rolling(window=w, min_periods=1).mean().reset_index(0, drop=True)
        df[f'waste_rate_rolling_mean_{w}'] = shifted_wr.groupby(df['product']).rolling(window=w, min_periods=1).mean().reset_index(0, drop=True)
        
    return df

def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates temperature dynamics based on temp_avg."""
    # Ensure types
    temp = df['temp_avg']
    shifted_temp = df.groupby('product')['temp_avg'].shift(1)
    
    df['temp_avg_lag_1'] = shifted_temp
    df['temp_avg_diff_1'] = temp - shifted_temp
    
    temp_7d_mean = shifted_temp.groupby(df['product']).rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
    df['temp_avg_vs_7d_mean'] = temp - temp_7d_mean
    
    df['is_cold_shock'] = (df['temp_avg_diff_1'] <= -5.0).astype(int)
    df['is_heat_shock'] = (df['temp_avg_diff_1'] >= 5.0).astype(int)
    
    return df

def build_operational_features(df_preprocessed: pd.DataFrame) -> pd.DataFrame:
    """
    Main orchestrator for feature engineering the operational table.
    """
    df = load_preprocessed_data(df_preprocessed)
    
    df = add_calendar_features(df)
    df = add_lag_features(df, lags=[1, 2, 3, 7, 14, 21, 28])
    df = add_rolling_features(df, windows=[7, 14, 28])
    df = add_same_weekday_features(df)
    df = add_ratio_features(df)
    df = add_temperature_features(df)
    
    # Convert string types to Categories for XGBoost
    df['weather'] = df['weather'].astype('category')
    df['product'] = df['product'].astype('category')
    
    return df

def get_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """
    Extracts purely predictive columns from the DataFrame.
    Strictly removes date, specific leakage vectors, and current-row targets.
    """
    # Core structural columns
    base_exclude = {'date', target_col}
    
    # Columns that inherently describe CURRENT day operational outcomes MUST BE REMOVED
    # to prevent target leakage regardless of what exactly we are forecasting.
    leakage_prone_operational_columns = {
        'sold_qty', 'waste_qty', 'made_qty', 'net_qty', 
        'gross_sales', 'waste_value',
        'has_positive_row', 'has_negative_row',
        'positive_row_count', 'negative_row_count', 'source_row_count',
        'only_negative_flag', 'multi_negative_rows_flag',
        'sell_through_rate', 'waste_rate'
    }
    
    # Identify final set
    features = []
    
    for col in df.columns:
        if col in base_exclude:
            continue
        if col in leakage_prone_operational_columns and col != target_col:
            continue
            
        features.append(col)
        
    return features
