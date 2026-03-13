import pytest
import pandas as pd
import numpy as np
from preprocess_history import build_daily_operational_table

@pytest.fixture
def raw_test_data() -> pd.DataFrame:
    """Creates mock raw transactional log rows with positive and negative quantities."""
    data = {
        'No.': [
            '2023-01-01', '2023-01-01', '2023-01-01', # Day 1: Mixed Croissant, Pos Baguette
            '2023-01-02', '2023-01-02', '2023-01-02', # Day 2: Multi-negative Croissant
            '2023-01-03' # Day 3: Only positive
        ],
        '상품명': [
            'Croissant', 'Croissant', 'Baguette',
            'Croissant', 'Croissant', 'Croissant',
            'Croissant'
        ],
        '수량': [
            50, -5, 30,  # 50 sold, 5 waste. 30 sold.
            -2, -3, 10,  # 10 sold, 5 total waste across 2 neg rows.
            15           # 15 sold.
        ],
        '실매출': [
            1000, -100, 600,
            -40, -60, 200,
            300
        ],
        'weather': ['Sunny', 'Sunny', 'Sunny', 'Rain', 'Rain', 'Rain', 'Cloudy'],
        'temp_avg': [20.0, 20.0, 20.0, 15.0, 15.0, 15.0, 18.0],
        'temp_max': [25.0, 25.0, 25.0, 18.0, 18.0, 18.0, 20.0],
        'temp_min': [15.0, 15.0, 15.0, 12.0, 12.0, 12.0, 16.0]
    }
    return pd.DataFrame(data)

def test_build_daily_operational_table(raw_test_data):
    df = build_daily_operational_table(raw_test_data)
    
    # Total unique date-product combos = 4
    assert len(df) == 4
    
    # Expected columns check
    expected_cols = [
         'date', 'product', 'sold_qty', 'waste_qty', 'made_qty', 'net_qty',
         'gross_sales', 'waste_value', 'weather', 'temp_avg', 
         'has_positive_row', 'has_negative_row', 'positive_row_count', 
         'negative_row_count', 'source_row_count', 'only_negative_flag', 
         'multi_negative_rows_flag'
    ]
    for col in expected_cols:
        assert col in df.columns
        
    # Check mixed group: Jan 1 Croissant
    jan1_croissant = df[(df['date'] == pd.to_datetime('2023-01-01')) & (df['product'] == 'Croissant')].iloc[0]
    assert jan1_croissant['sold_qty'] == 50
    assert jan1_croissant['waste_qty'] == 5 # absolute value
    assert jan1_croissant['made_qty'] == 55 # 50 + 5
    assert jan1_croissant['net_qty'] == 45 # 50 - 5
    assert jan1_croissant['gross_sales'] == 1000
    assert jan1_croissant['waste_value'] == 100
    assert jan1_croissant['source_row_count'] == 2
    assert jan1_croissant['has_negative_row'] == 1
    assert jan1_croissant['only_negative_flag'] == 0
    assert jan1_croissant['multi_negative_rows_flag'] == 0
    
    # Check multi-negative group: Jan 2 Croissant
    jan2_croissant = df[(df['date'] == pd.to_datetime('2023-01-02')) & (df['product'] == 'Croissant')].iloc[0]
    assert jan2_croissant['sold_qty'] == 10
    assert jan2_croissant['waste_qty'] == 5 # abs(-2 + -3)
    assert jan2_croissant['source_row_count'] == 3
    assert jan2_croissant['negative_row_count'] == 2
    assert jan2_croissant['multi_negative_rows_flag'] == 1

def test_negative_only_group():
    data = {
        'No.': ['2023-01-01'],
        '상품명': ['Croissant'],
        '수량': [-10],
        '실매출': [-200]
    }
    df_raw = pd.DataFrame(data)
    df = build_daily_operational_table(df_raw)
    
    assert len(df) == 1
    row = df.iloc[0]
    assert row['sold_qty'] == 0
    assert row['waste_qty'] == 10
    assert row['made_qty'] == 10
    assert row['net_qty'] == -10
    assert row['only_negative_flag'] == 1
    assert row['has_positive_row'] == 0

def test_missing_weather_cols():
    # Only supply mandatory columns
    data = {
        'No.': ['2023-01-01'],
        '상품명': ['Croissant'],
        '수량': [10]
    }
    df_raw = pd.DataFrame(data)
    df = build_daily_operational_table(df_raw)
    
    # Weather columns should be generated with missing values smoothly
    assert 'weather' in df.columns
    assert 'temp_min' in df.columns
    assert pd.isna(df.iloc[0]['temp_max'])
