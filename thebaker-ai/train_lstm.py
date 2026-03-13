"""
train_lstm.py - Challenger model training using PyTorch LSTM.

Responsible for training an LSTM sequence model for operational product demand.
Uses a global model across all products, taking historical sequences of
quantities and temperatures, and appending known-future contexts like weather and 
calendar dynamics to predict specific operational targets without leakage.

Targets (e.g., 'sold_qty', 'waste_qty') are treated independently, saving separate
model artifacts for the deterministic ensemble orchestrator.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from sklearn.preprocessing import StandardScaler
import joblib

# Ensure reproducibility where possible
torch.manual_seed(42)

def calculate_lstm_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MAE and WAPE."""
    mae = np.mean(np.abs(y_true - y_pred))
    sum_actuals = np.sum(y_true)
    wape = np.sum(np.abs(y_true - y_pred)) / sum_actuals if sum_actuals > 0 else 0.0
    return {'mae': mae, 'wape': wape}

# ---------------------------------------------------------------------------
# Data Preparation

class BakeryLSTMDataset(Dataset):
    """
    PyTorch Dataset for Bakery operational sequence data.
    
    Predicts day T using:
    - Historic Sequence (T-seq_len to T-1): core operations + cyclic metrics.
    - Target Day Context (T): expected weather, expected temp, calendar constraints.
    """
    def __init__(self, df: pd.DataFrame, target_col: str, seq_len: int,
                 historic_continuous_cols: List[str],
                 future_continuous_cols: List[str],
                 product_mapping: Dict[str, int],
                 weather_mapping: Dict[str, int]):
        self.seq_len = seq_len
        self.historic_continuous_cols = historic_continuous_cols
        self.future_continuous_cols = future_continuous_cols
        
        self.X_hist_num = []
        self.X_target_num = []
        self.X_prod = []
        self.X_target_weather = []
        self.y = []
        
        df = df.copy()
        # Strictly chronological per product to form unbroken historical sequences
        df = df.sort_values(by=['product', 'date']).reset_index(drop=True)
        
        for prod, group in df.groupby('product', observed=False):
            if len(group) <= seq_len:
                continue
                
            hist_vals = group[historic_continuous_cols].values
            fut_vals = group[future_continuous_cols].values
            
            prod_val = product_mapping.get(prod, 0)
            weather_vals = np.array([weather_mapping.get(w, 0) for w in group['weather']])
            
            targets = group[target_col].values
            
            # Sliding window 
            for i in range(len(group) - seq_len):
                # sequence indices: [i, i+seq_len-1]
                # target index: i+seq_len
                
                # History (T-seq_len to T-1)
                self.X_hist_num.append(hist_vals[i : i+seq_len])
                
                # Known Target day Context (T)
                target_idx = i + seq_len
                self.X_target_num.append(fut_vals[target_idx])
                
                self.X_prod.append(prod_val)
                self.X_target_weather.append(weather_vals[target_idx])
                
                self.y.append(targets[target_idx])
                
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_hist_num[idx], dtype=torch.float32),      # (Seq, Hist_Feat)
            torch.tensor(self.X_target_num[idx], dtype=torch.float32),    # (Fut_Feat)
            torch.tensor(self.X_prod[idx], dtype=torch.long),             # Scalar
            torch.tensor(self.X_target_weather[idx], dtype=torch.long),   # Scalar
            torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(-1)  # (1)
        )

def prepare_lstm_data(df: pd.DataFrame, target_col: str, seq_len: int = 28,
                      scaler_hist: Optional[StandardScaler] = None,
                      scaler_fut: Optional[StandardScaler] = None,
                      product_map: Optional[Dict[str, int]] = None,
                      weather_map: Optional[Dict[str, int]] = None
                      ) -> Tuple[BakeryLSTMDataset, StandardScaler, StandardScaler, dict, dict]:
    """
    Orchestrates continuous scaling, categorical mappings, and dataset generation.
    Returns (dataset, hist_scaler, fut_scaler, product_map, weather_map).
    """
    df = df.copy()
    
    # Define exact sequence requirements
    # History includes everything operational + temporal
    historic_continuous_cols = [
        'sold_qty', 'waste_qty', 'made_qty', 'temp_avg', 
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
    ]
    # Future/Target day known context
    future_continuous_cols = [
        'temp_avg', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
    ]
    
    # Safeguard if missing
    for c in historic_continuous_cols + future_continuous_cols:
        if c not in df.columns:
            df[c] = 0.0
            
    # Fit/Transform scalers
    if scaler_hist is None:
        scaler_hist = StandardScaler()
        df[historic_continuous_cols] = scaler_hist.fit_transform(df[historic_continuous_cols])
    else:
        df[historic_continuous_cols] = scaler_hist.transform(df[historic_continuous_cols])
        
    if scaler_fut is None:
        scaler_fut = StandardScaler()
        df[future_continuous_cols] = scaler_fut.fit_transform(df[future_continuous_cols])
    else:
        df[future_continuous_cols] = scaler_fut.transform(df[future_continuous_cols])
        
    # Maps for embedded categories
    if product_map is None:
        unique_prods = df['product'].unique()
        product_map = {prod: i+1 for i, prod in enumerate(unique_prods)}
        product_map['<UNNK>'] = 0 
        
    if weather_map is None:
        unique_weather = df['weather'].dropna().unique()
        weather_map = {w: i+1 for i, w in enumerate(unique_weather)}
        weather_map['<UNNK>'] = 0 

    dataset = BakeryLSTMDataset(
        df, target_col, seq_len, 
        historic_continuous_cols, future_continuous_cols, 
        product_map, weather_map
    )
    
    return dataset, scaler_hist, scaler_fut, product_map, weather_map

# ---------------------------------------------------------------------------
# Model Architecture

class BakeryLSTM(nn.Module):
    """
    Challenger LSTM capturing chronological history and target-context constraints.
    """
    def __init__(self, 
                 num_hist_features: int,
                 num_fut_features: int,
                 num_products: int, 
                 num_weather: int,
                 prod_emb_dim: int = 8,
                 weather_emb_dim: int = 4,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super(BakeryLSTM, self).__init__()
        
        self.prod_embedding = nn.Embedding(num_products + 1, prod_emb_dim) # Unknown handling
        self.weather_embedding = nn.Embedding(num_weather + 1, weather_emb_dim)
        
        # Sequence processing
        self.lstm = nn.LSTM(
            input_size=num_hist_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Concat: LSTM last hidden + Prod Emb + Weather Emb + Future Temps/Calendar
        dense_in_features = hidden_dim + prod_emb_dim + weather_emb_dim + num_fut_features
        
        self.fc = nn.Sequential(
            nn.Linear(dense_in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_hist_num, x_fut_num, x_prod, x_target_weather):
        """
        x_hist: (B, Seq, HistFeat)
        x_fut:  (B, FutFeat)
        x_prod: (B)
        x_weath:(B)
        """
        out, (hn, cn) = self.lstm(x_hist_num)
        last_out = out[:, -1, :] # (B, Hidden)
        
        prod_emb = self.prod_embedding(x_prod)           # (B, ProdEmb)
        weath_emb = self.weather_embedding(x_target_weather) # (B, WeaEmb)
        
        combined = torch.cat([last_out, prod_emb, weath_emb, x_fut_num], dim=1) # (B, DenseIn)
        
        prediction = self.fc(combined) # (B, 1)
        return prediction

# ---------------------------------------------------------------------------
# Training Orchestration

def train_lstm_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, 
                     criterion: nn.Module, device: torch.device, clip_val: float = 1.0) -> float:
    """Trains one epoch."""
    model.train()
    total_loss = 0.0
    
    for x_h, x_f, x_p, x_tw, y in loader:
        x_h, x_f, x_p, x_tw, y = x_h.to(device), x_f.to(device), x_p.to(device), x_tw.to(device), y.to(device)
        
        optimizer.zero_grad()
        preds = model(x_h, x_f, x_p, x_tw)
        loss = criterion(preds, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()
        
        total_loss += loss.item() * x_h.size(0)
        
    return total_loss / len(loader.dataset)

def validate_lstm(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Runs validation enforcing non-negative predictions."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_h, x_f, x_p, x_tw, y in loader:
            x_h, x_f, x_p, x_tw, y = x_h.to(device), x_f.to(device), x_p.to(device), x_tw.to(device), y.to(device)
            preds = model(x_h, x_f, x_p, x_tw)
            
            preds = torch.clamp(preds, min=0.0) # Constraint logic built natively into the validation loop
            loss = criterion(preds, y)
            total_loss += loss.item() * x_h.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    val_loss = total_loss / len(loader.dataset)
    metrics = calculate_lstm_metrics(np.array(all_targets).flatten(), np.array(all_preds).flatten())
    
    return val_loss, metrics

def train_lstm_model(train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None,
                     target_col: str = 'sold_qty', seq_len: int = 28,
                     epochs: int = 30, batch_size: int = 32, lr: float = 1e-3,
                     patience: int = 5) -> Tuple[BakeryLSTM, StandardScaler, StandardScaler, dict, dict, Dict[str, Any]]:
    """
    Trains the sequence challenger targeting operational quantities with early stopping.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset, s_hist, s_fut, p_map, w_map = prepare_lstm_data(train_df, target_col, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_df is not None:
        val_dataset, _, _, _, _ = prepare_lstm_data(val_df, target_col, seq_len, s_hist, s_fut, p_map, w_map)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
        
    model = BakeryLSTM(
        num_hist_features=len(train_dataset.historic_continuous_cols),
        num_fut_features=len(train_dataset.future_continuous_cols),
        num_products=len(p_map),
        num_weather=len(w_map),
        hidden_dim=64,
        num_layers=1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() 
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    
    for epoch in range(epochs):
        train_loss = train_lstm_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        
        if val_loader:
            val_loss, metrics = validate_lstm(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(metrics['mae'])
            
            print(f"Epoch {epoch+1:02d}/{epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val MAE: {metrics['mae']:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train MSE: {train_loss:.4f}")
            best_model_state = model.state_dict()
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, s_hist, s_fut, p_map, w_map, history

# ---------------------------------------------------------------------------
# Utilities and I/O

def save_lstm_artifacts(model: BakeryLSTM, s_hist: StandardScaler, s_fut: StandardScaler, 
                        p_map: dict, w_map: dict, target_col: str, base_dir: str = 'models/'):
    """Saves torch states targeting specific operational outputs explicitly."""
    os.makedirs(base_dir, exist_ok=True)
    
    prefix = f"lstm_{target_col}"
    torch.save(model.state_dict(), os.path.join(base_dir, f'{prefix}_weights.pt'))
    joblib.dump(s_hist, os.path.join(base_dir, f'{prefix}_scaler_hist.pkl'))
    joblib.dump(s_fut, os.path.join(base_dir, f'{prefix}_scaler_fut.pkl'))
    joblib.dump(p_map, os.path.join(base_dir, f'{prefix}_pmap.pkl'))
    joblib.dump(w_map, os.path.join(base_dir, f'{prefix}_wmap.pkl'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Challenger model on preprocessed table.")
    parser.add_argument("--target", "-t", type=str, default="waste_qty", 
                        choices=["sold_qty", "waste_qty"], 
                        help="Operational target column (default: waste_qty)")
    parser.add_argument("--input", "-i", type=str, default="features_clean.csv", 
                        help="Path to precomputed features CSV")
    
    args = parser.parse_args()
    
    print(f"[{args.target}] Initializing LSTM sequence training...")
    
    if os.path.exists(args.input):
        df = pd.read_csv(args.input)
        df['date'] = pd.to_datetime(df['date'])
    else:
        print(f"Warning: {args.input} not found. Generating sequence dummy data.")
        dates = pd.date_range("2023-01-01", "2023-04-30")
        products = ['Croissant', 'Baguette']
        
        rows = []
        for d in dates:
            for p in products:
                rows.append({
                    'date': d.strftime("%Y-%m-%d"),
                    'product': p,
                    'sold_qty': int(np.random.normal(50, 10)),
                    'waste_qty': int(np.random.normal(5, 2)),
                    'made_qty': 60.0,
                    'weather': 'Sunny' if np.random.rand() > 0.3 else 'Rainy',
                    'temp_avg': 20.0,
                    'dow_sin': np.sin(2 * np.pi * d.dayofweek / 7),
                    'dow_cos': np.cos(2 * np.pi * d.dayofweek / 7),
                    'month_sin': 0.5,
                    'month_cos': 0.5
                })
        df = pd.DataFrame(rows)
    
    # Needs chronological sequence boundaries
    df['date'] = pd.to_datetime(df['date'])
    max_date = df['date'].max()
    split_date = max_date - pd.Timedelta(days=21)
    
    train_df = df[df['date'] < split_date]
    val_df = df[df['date'] >= split_date]
    
    print(f"Splitting data temporally at {split_date}. Sequence length = 14 days.")
    
    model, s_h, s_f, p_map, w_map, hist = train_lstm_model(
        train_df, val_df, 
        target_col=args.target, 
        seq_len=14, 
        epochs=15,
        batch_size=16
    )
    
    print(f"\n[{args.target}] Final Validation MAE: {hist['val_mae'][-1]:.2f}")
    
    print(f"Saving {args.target} LSTM artifacts to models/...")
    save_lstm_artifacts(model, s_h, s_f, p_map, w_map, target_col=args.target)
    print("Done.")
