import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_dummy_history(output_file="history.csv", days=365):
    """
    Generates a realistic-looking fake bakery dataset for a public portfolio.
    Replaces real business numbers with mathematically consistent simulations.
    """
    products = ['Croissant', 'Baguette', 'Sourdough Loaf', 'Glazed Donut']
    weathers = ['Sunny', 'Cloudy', 'Rainy', 'Snow']
    
    start_date = datetime.now() - timedelta(days=days)
    records = []
    
    print(f"Generating {days} days of simulated bakery logs...")
    
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        
        # Simulate seasonal weather
        month = current_date.month
        base_temp = 25 if month in [6,7,8] else (5 if month in [12,1,2] else 15)
        temp_actual = round(base_temp + np.random.normal(0, 5), 1)
        
        weather = 'Sunny'
        if temp_actual < 0: weather = 'Snow'
        elif np.random.random() < 0.3: weather = 'Rainy'
        elif np.random.random() < 0.5: weather = 'Cloudy'
        
        for product in products:
            # Base demand shifts by product
            base_demand = 50 if product == 'Croissant' else 30
            
            # Weekend bump
            if current_date.weekday() >= 5:
                base_demand = int(base_demand * 1.5)
                
            # Weather impact
            if weather == 'Rainy' and product == 'Glazed Donut':
                base_demand = int(base_demand * 0.7)
            
            # Simulate a day's true sales
            sold = max(5, int(np.random.normal(base_demand, 10)))
            
            # Simulate waste (baker made slightly too much)
            waste = -1 * abs(int(np.random.normal(5, 3)))
            
            # Create discrete ledger entries for sold items
            # (In reality this is many rows, but we aggregate slightly for file size)
            if sold > 0:
                records.append({
                    'No.': current_date.strftime("%Y-%m-%d"),
                    '상품명': product,
                    '수량': sold,
                    '실매출': sold * 3500, # Fake KRW price
                    'weather': weather,
                    'temp_avg': temp_actual,
                    'temp_max': round(temp_actual + 5, 1),
                    'temp_min': round(temp_actual - 5, 1)
                })
                
            # Create discrete ledger entries for end of day waste
            if waste < 0:
                records.append({
                    'No.': current_date.strftime("%Y-%m-%d"),
                    '상품명': product,
                    '수량': waste,
                    '실매출': 0,
                    'weather': weather,
                    'temp_avg': temp_actual,
                    'temp_max': round(temp_actual + 5, 1),
                    'temp_min': round(temp_actual - 5, 1)
                })
                
    df = pd.DataFrame(records)
    
    # Shuffle slightly to mimic real disorganized POS logs
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"Successfully wrote {len(df)} fake records to {output_file}. Safe for public GitHub.")

if __name__ == "__main__":
    generate_dummy_history()
