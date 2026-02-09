import os
import pandas as pd
import json
from google import genai
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

# --- 1. SETUP ---
app = FastAPI()

# Load API Key
load_dotenv()
GENAI_KEY = os.getenv("GOOGLE_API_KEY")

if not GENAI_KEY:
    raise ValueError("No API Key found! Please check your .env file.")

# Initialize the NEW Client
client = genai.Client(api_key=GENAI_KEY)

# Point to your CSV
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'main', 'resources', 'history.csv')

# Load Data
try:
    df = pd.read_csv(CSV_PATH)
    df.columns = ['date', 'product', 'qty', 'weather', 'temp']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print("✅ Bakery Data Loaded Successfully")
except Exception as e:
    print(f"⚠️ Error loading CSV: {e}")
    df = pd.DataFrame()

# --- 2. DATA MODELS ---
class PredictionRequest(BaseModel):
    product: str
    target_date: str
    weather_forecast: str
    temp_forecast: int

class PredictionResponse(BaseModel):
    prediction: int
    reasoning: str

# --- 3. HELPER FUNCTIONS ---
def get_historical_context(product, target_date_str):
    if df.empty: return "No historical data available."
    
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format."

    day_name = target_date.strftime("%A")
    
    # Filter: Same Product + Same Day of Week
    history = df[
        (df['product'] == product) & 
        (df['date'].dt.day_name() == day_name)
    ].sort_values(by='date', ascending=False).head(5)
    
    if history.empty:
        return f"No history found for {product} on {day_name}s."
    
    context_str = f"Sales history for {product} on previous {day_name}s:\n"
    for _, row in history.iterrows():
        date_str = row['date'].strftime("%Y-%m-%d")
        context_str += f"- {date_str}: Sold {row['qty']} units (Weather: {row['weather']}, {row['temp']}°C)\n"
    
    return context_str

# --- 4. THE ENDPOINT ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_sales(request: PredictionRequest):
    # A. Get Context
    context = get_historical_context(request.product, request.target_date)
    
    # B. Construct Prompt
    prompt = f"""
    You are an expert bakery forecaster.
    TASK: Predict sales for '{request.product}' on {request.target_date}.
    
    CONTEXT:
    1. Forecast: {request.weather_forecast}, {request.temp_forecast}°C.
    2. {context}
    
    RULES:
    - Return ONLY valid JSON.
    - Format: {{ "prediction": 15, "reasoning": "..." }}
    """
    
    # C. Call Gemini (NEW SYNTAX)
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        # Clean response text
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3]
        
        data = json.loads(text)
        
        return PredictionResponse(
            prediction=data.get("prediction", 0),
            reasoning=data.get("reasoning", "AI generated based on trends.")
        )
    
    except Exception as e:
        print(f"LLM Error: {e}")
        return PredictionResponse(prediction=10, reasoning=f"Fallback error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)