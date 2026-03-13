"""
preprocess_history.py - Raw CSV log preprocessor for The Baker V3.

Responsible for safely loading the raw event log, renaming Korean columns, parsing dates,
and aggregating the signed quantities into a daily product-level operational table.
This unified table serves as the source of truth for downstream features.py and models.
"""

import pandas as pd
import numpy as np
import argparse
from typing import Optional

def build_daily_operational_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a dataframe of raw transactional event logs into a clean, 
    daily product-level operational dataset. 
    
    Expected Input DataFrame Columns (or subset if sales/weather missing):
    - No. (Date)
    - 상품명 (Product)
    - 수량 (Signed Qty)
    - 실매출 (Signed Sales)
    - weather
    - temp_avg
    - temp_max
    - temp_min
    
    Output specifies detailed sold/waste aggregations per date-product.
    """
    df = df_raw.copy()
    
    # 1. Rename core columns if they exist in the raw schema
    rename_map = {
        'No.': 'date',
        '상품명': 'product',
        '수량': 'qty',
        '실매출': 'sales'
    }
    df = df.rename(columns=rename_map)
    
    # Ensure mandatory columns exist
    if 'date' not in df.columns or 'product' not in df.columns or 'qty' not in df.columns:
        raise ValueError("Missing essential columns (No., 상품명, 수량) in raw dataset.")
        
    # 2. Parse Date and Numeric values safely
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(0)
    
    if 'sales' in df.columns:
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0)
    else:
        df['sales'] = 0.0

    # Ensure weather/temp columns exist as mock arrays if missing from source to prevent crashes
    for col in ['weather', 'temp_avg', 'temp_max', 'temp_min']:
        if col not in df.columns:
            df[col] = np.nan if col.startswith('temp') else None
    
    # 3. Intermediate Row-Level Calculations
    df['is_positive'] = df['qty'] > 0
    df['is_negative'] = df['qty'] < 0
    
    df['positive_qty'] = np.where(df['is_positive'], df['qty'], 0)
    df['negative_qty'] = np.where(df['is_negative'], df['qty'], 0)
    
    df['positive_sales'] = np.where(df['is_positive'], df['sales'], 0)
    df['negative_sales'] = np.where(df['is_negative'], df['sales'], 0)
    
    # 4. Group by Date and Product
    grouped = df.groupby(['date', 'product'], dropna=False)
    
    agg_funcs = {
        'qty': ['count'], # source_row_count
        'is_positive': ['sum', 'max'], # count of pos rows, has_positive_row
        'is_negative': ['sum', 'max'], # count of neg rows, has_negative_row
        
        'positive_qty': ['sum'], # sold_qty
        'negative_qty': ['sum'], # negative waste_qty (need abs later)
        
        'positive_sales': ['sum'], # gross_sales
        'negative_sales': ['sum'], # negative waste_value (need abs later)
        
        # Weather / Temperature - take first valid observation
        'weather': ['first'],
        'temp_avg': ['first'],
        'temp_max': ['first'],
        'temp_min': ['first']
    }
    
    res = grouped.agg(agg_funcs).reset_index()
    
    # Flatten multi-level columns
    res.columns = [
        'date', 'product', 
        'source_row_count', 
        'positive_row_count', 'has_positive_row', 
        'negative_row_count', 'has_negative_row',
        'sold_qty', 'raw_waste_qty',
        'gross_sales', 'raw_waste_value',
        'weather', 'temp_avg', 'temp_max', 'temp_min'
    ]
    
    # Fix boolean flags
    res['has_positive_row'] = res['has_positive_row'].astype(int)
    res['has_negative_row'] = res['has_negative_row'].astype(int)
    res['positive_row_count'] = res['positive_row_count'].astype(int)
    res['negative_row_count'] = res['negative_row_count'].astype(int)
    
    # 5. Business Logic Calculations
    res['sold_qty'] = res['sold_qty']
    res['waste_qty'] = res['raw_waste_qty'].abs()
    res['made_qty'] = res['sold_qty'] + res['waste_qty']
    res['net_qty'] = res['sold_qty'] - res['waste_qty']
    
    res['gross_sales'] = res['gross_sales']
    res['waste_value'] = res['raw_waste_value'].abs()
    
    # Logic flags
    res['only_negative_flag'] = ((res['has_negative_row'] == 1) & (res['has_positive_row'] == 0)).astype(int)
    res['multi_negative_rows_flag'] = (res['negative_row_count'] > 1).astype(int)
    
    # Cleanup intermediate columns
    res = res.drop(columns=['raw_waste_qty', 'raw_waste_value'])
    
    # Ensure ordered and typed
    cols_order = [
         'date', 'product', 
         'sold_qty', 'waste_qty', 'made_qty', 'net_qty',
         'gross_sales', 'waste_value',
         'weather', 'temp_avg', 'temp_max', 'temp_min',
         'has_positive_row', 'has_negative_row',
         'positive_row_count', 'negative_row_count', 'source_row_count',
         'only_negative_flag', 'multi_negative_rows_flag'
    ]
    
    # Handle NaTs cleanly
    res = res.dropna(subset=['date', 'product'])
    
    return res[cols_order].sort_values(['date', 'product']).reset_index(drop=True)

def process_file(input_path: str, output_path: str) -> None:
    """Reads raw CSV from disk, processes it, and saves to Output CSV."""
    print(f"Loading raw data from {input_path}...")
    try:
        df_raw = pd.read_csv(input_path)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return
        
    df_clean = build_daily_operational_table(df_raw)
    
    print(f"Processed into {len(df_clean)} operational rows.")
    
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess The Baker V3 raw history CSV into an operational table.")
    parser.add_argument("--input", "-i", type=str, default="history.csv", help="Path to raw CSV (default: history.csv)")
    parser.add_argument("--output", "-o", type=str, default="history_clean.csv", help="Path to output CSV (default: history_clean.csv)")
    
    args = parser.parse_args()
    process_file(args.input, args.output)
