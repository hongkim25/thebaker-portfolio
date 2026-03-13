"""
api.py - FastAPI service for The Baker V3 operational forecasting.

Exposes a thin web layer that:
1. Validates incoming JSON requests for specific product/date forecasts.
2. Converts payloads into pandas DataFrames.
3. Delegates business logic to the `forecast_graph.py` LangGraph orchestrator.
4. Returns the structured explanation and dual-target numerical outputs.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import pandas as pd
import uvicorn
import datetime

# Import the core LangGraph orchestrator
from forecast_graph import execute_forecast_workflow

app = FastAPI(
    title="The Baker V3 Operational API",
    description="Deterministic demand forecasting and production planning.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow local React client requests
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------

class ForecastRequest(BaseModel):
    product: str = Field(..., example="Croissant")
    target_date: str = Field(..., example="2023-11-20")
    forecast_weather: str = Field(..., example="Rainy")
    forecast_temp: float = Field(..., example=12.5)
    forecast_temp_max: Optional[float] = Field(None, example=15.0)
    forecast_temp_min: Optional[float] = Field(None, example=10.0)

class ForecastResponse(BaseModel):
    product: str
    target_date: str
    
    # Mathematical Outputs
    predicted_sold_qty: float
    predicted_waste_qty: float
    recommended_made_qty: float
    
    # Anomaly tracking
    sold_anomaly_flag: bool
    waste_anomaly_flag: bool
    
    # Explanation
    explanation: str
    risk_note: str
    confidence_label: str
    production_note: str
    
    # Raw Pipeline State (Optional metadata block)
    metadata: Dict[str, Any]

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    """Simple availability ping."""
    return {"status": "ok", "service": "The Baker V3"}

@app.post("/forecast", response_model=ForecastResponse)
def generate_forecast(request: ForecastRequest):
    """
    Main operational endpoint.
    Translates REST payload into mathematical tensors, queries LangGraph orchestrator,
    and formats final staff-ready outputs.
    """
    try:
        # Validate date visually
        datetime.date.fromisoformat(request.target_date)
        
        # In a real environment, we would hit a database here to pull the historical `request_features_df`
        # for this specific `product` leading up to `target_date`.
        # For this thin inference API, we construct a mock feature slice representing today.
        
        feature_dict = {
            'date': [request.target_date],
            'product': [request.product],
            'weather': [request.forecast_weather],
            'temp_avg': [request.forecast_temp]
        }
        
        if request.forecast_temp_max is not None:
            feature_dict['temp_max'] = [request.forecast_temp_max]
        if request.forecast_temp_min is not None:
            feature_dict['temp_min'] = [request.forecast_temp_min]
            
        df_features = pd.DataFrame(feature_dict)
        
        # Execute the LangGraph workflow directly
        graph_result = execute_forecast_workflow(
            product=request.product,
            target_date=request.target_date,
            feature_df=df_features
        )
        
        # In cases where the graph completely fails logic checks (empty DF, missing nodes)
        if "error" in graph_result.get('explanation', '').lower() or graph_result.get('confidence_label') == 'Failed':
            raise ValueError("LangGraph execution encountered an operational failure.")
        
        # Mute potential Nones explicitly to 0.0 for safety serialization
        sold_qty = float(graph_result.get('sold_final_prediction') or 0.0)
        waste_qty = float(graph_result.get('waste_final_prediction') or 0.0)
        made_qty = float(graph_result.get('recommended_made_qty') or 0.0)
        
        return ForecastResponse(
            product=graph_result['product'],
            target_date=graph_result['target_date'],
            predicted_sold_qty=sold_qty,
            predicted_waste_qty=waste_qty,
            recommended_made_qty=made_qty,
            sold_anomaly_flag=graph_result.get('sold_anomaly_flag', False),
            waste_anomaly_flag=graph_result.get('waste_anomaly_flag', False),
            explanation=graph_result.get('explanation', ''),
            risk_note=graph_result.get('risk_note', ''),
            confidence_label=graph_result.get('confidence_label', ''),
            production_note=graph_result.get('production_note', ''),
            metadata={
                "sold_xgb": graph_result.get('sold_xgb_prediction'),
                "sold_lstm": graph_result.get('sold_lstm_prediction'),
                "sold_gap": graph_result.get('sold_gap_ratio'),
                "waste_xgb": graph_result.get('waste_xgb_prediction'),
                "waste_lstm": graph_result.get('waste_lstm_prediction'),
                "waste_gap": graph_result.get('waste_gap_ratio'),
                "selected_strategies": graph_result.get('selected_strategies')
            }
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# ---------------------------------------------------------
# Local Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Starting The Baker V3 operational API on port 8000...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
