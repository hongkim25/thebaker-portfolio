"""
explain.py - Natural language explanation layer for The Baker V3.

Responsible for taking mathematically validated forecasting outputs
(sold_qty, waste_qty, recommended_made_qty) and wrapping them in
plain English explanations via an LLM.

Strict Rule: The LLM MUST NEVER generate or modify the numerical forecast.
It solely acts as an interpreter explaining the supplied context.
"""

import os
import json
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Structured Schemas
# ---------------------------------------------------------

class ForecastContext(BaseModel):
    """The strictly typed, mathematically validated input context."""
    product: str
    target_date: str
    forecast_weather: str
    forecast_temp: float
    predicted_sold_qty: float
    predicted_waste_qty: float
    recommended_made_qty: float
    anomaly_flag: bool = False
    key_drivers: Optional[str] = None

class ExplanationOutput(BaseModel):
    """The required JSON structure from the LLM."""
    staff_explanation: str = Field(description="A friendly, 2-3 sentence explanation of the forecast and production recommendation.")
    risk_note: str = Field(description="A short note on expected waste, shortages, or weather risks.")
    confidence_label: str = Field(description="One of: 'High', 'Moderate', 'Low'.")
    production_note: str = Field(description="Operational summary matching the exact recommended_made_qty.")

# ---------------------------------------------------------
# Core Logic
# ---------------------------------------------------------

def load_gemini() -> genai.GenerativeModel:
    """Initializes the Gemini client if an API key is available."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-generativeai is not installed.")
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")
        
    genai.configure(api_key=api_key)
    
    # Use Pro model for reasoning, enforcing strict JSON schema output
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ExplanationOutput,
            temperature=0.2 # low temp to prevent hallucination
        )
    )
    return model

def build_prompt(context: ForecastContext) -> str:
    """Constructs the prompt, stringently enforcing the No-Math rule."""
    
    anomaly_text = "YES. The prediction tracking systems detected unstable pattern divergence today." if context.anomaly_flag else "None detected."
    driver_text = f"Key indicators: {context.key_drivers}" if context.key_drivers else "Relying on standard historical averages."
    
    return f"""
You are the assistant for The Baker V3 operational system.
Your job is to explain the daily production forecast to the bakery staff.

CRITICAL RULES:
1. YOU MUST NEVER CHANGE THE NUMBERS PROVIDED TO YOU.
2. The numbers are ALREADY CALCULATED by a deterministic machine learning system. You are just explaining them.
3. Use friendly, clear language appropriate for store employees.

--- VALIDATED SYSTEM OUTPUTS ---
Product: {context.product}
Target Date: {context.target_date}

Weather Forecast: {context.forecast_weather}
Temperature: {context.forecast_temp}°C

Predicted Demand (Sold): {context.predicted_sold_qty:.1f}
Predicted Waste (Leftovers): {context.predicted_waste_qty:.1f}
Total Recommended to Make: {context.recommended_made_qty:.1f}

System Anomalies: {anomaly_text}
{driver_text}
--------------------------------

Generate the explanation matching the required JSON structure.
Match your `production_note` output EXACTLY to the "Total Recommended to Make" number provided.
"""

def generate_fallback_explanation(context: ForecastContext) -> Dict[str, Any]:
    """Provides a safe, mathematically accurate layout if the LLM crashes."""
    
    anomaly_warning = " (Note: System anomaly flagged, numbers fall back to primary averages.)" if context.anomaly_flag else ""
    
    return {
        "staff_explanation": f"For {context.target_date}, the system predicts we will sell {context.predicted_sold_qty:.0f} {context.product}s. Weather is expected to be {context.forecast_weather} ({context.forecast_temp}°C).{anomaly_warning}",
        "risk_note": f"Estimated buffer waste is {context.predicted_waste_qty:.0f} units.",
        "confidence_label": "Moderate (Fallback Gen)",
        "production_note": f"Please prepare {context.recommended_made_qty:.0f} units."
    }

def generate_explanation(context: ForecastContext) -> Dict[str, Any]:
    """
    Main orchestrator handling the context-to-LLM pipeline.
    Catches ALL errors and silently returns the fallback template.
    """
    prompt = build_prompt(context)
    
    try:
        model = load_gemini()
        response = model.generate_content(prompt)
        # Parse the guaranteed JSON
        result = json.loads(response.text)
        return result
    except Exception as e:
        print(f"Warning: LLM generation failed ({e}). Using deterministic fallback.")
        return generate_fallback_explanation(context)

if __name__ == "__main__":
    # Test execution / Demo
    test_context = ForecastContext(
        product="Croissant",
        target_date="2023-11-20",
        forecast_weather="Rainy",
        forecast_temp=12.5,
        predicted_sold_qty=45.0,
        predicted_waste_qty=5.0,
        recommended_made_qty=50.0,
        anomaly_flag=False,
        key_drivers="Historical moving average is 48. Rain typically drops sales by 5%."
    )
    
    print("--- Testing Context ---")
    print(test_context.model_dump_json(indent=2))
    print("\n--- Generating Explanation ---")
    
    os.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', 'dummy_key')
    
    output = generate_explanation(test_context)
    print(json.dumps(output, indent=2))
