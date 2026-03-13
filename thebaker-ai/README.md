# The Baker V3 - Operations Forecasting Ecosystem

**An enterprise-grade, deterministic demand forecasting and production orchestration platform designed specifically for live bakery operations.**

---

## 1. Project Summary
The Baker V3 is an end-to-end Applied AI ecosystem that predicts daily bakery demand, quantifies anticipated waste, and generates human-readable production recommendations for bakery staff. It bridges the gap between raw, messy transactional logs and actionable, intuitive storefront decisions. 

Because bakery forecasting is highly sensitive to weather, day-of-week seasonality, and product-specific lifecycles, V3 abandons simplistic moving averages in favor of an orchestrated ensemble of dual-target Machine Learning models (predicting both *sales* and *waste* simultaneously). The entire backend is hidden behind a polished, intuitive React frontend designed for non-technical bakery operators to use daily.

This repository serves as a showcase of:
- **For Recruiters / Hiring Managers:** A complete, production-ready product lifecycle—from raw data ingestion to a beautiful user interface—proving the ability to build, deploy, and explain complex AI systems that solve real-world unit economics.
- **For Engineers / Reviewers:** Strict architectural separation of concerns. Model training is isolated from online inference. Fallback heuristics and deterministic ensembles guard against neural hallucination. Language models (`Gemini`) are strictly isolated to explanation generation, never numeric prediction.

## 2. Problem Statement
Bakery production planning is fundamentally difficult:
- **Overproduction creates waste:** Unsold perishable goods immediately destroy profit margins at the end of the day.
- **Underproduction loses sales:** Empty shelves lead to missed revenue and dissatisfied customers.
- **Weather Sensitivity:** Rain, snow, and temperature shocks drastically alter walk-in traffic and specific product affinity.

Operators need a scientific way to balance the risk of waste against the upside of sales without needing a data science degree to interpret the outcome.

## 3. Why Raw Bakery Logs Were Difficult to Model
Historically, "The Baker" tracked inventory through a single signed quantity (`qty`) column. 
- Positive numbers represented production (+40).
- Negative numbers represented a mix of *sales* (-30) and end-of-day *waste* (-10). 

This conflation made direct forecasting impossible. An AI model couldn't tell if a "-10" on a rainy day meant the product was incredibly popular (sold out) or completely unpopular (thrown away).

## 4. Preprocessing Layer
To fix this, the system pipeline begins with `preprocess_history.py`. This layer:
- Ingests the messy raw transactional logs.
- Isolates positive inputs to derive `made_qty`.
- Isolates negative inputs and uses strict temporal row-counting heuristics to separate legitimate `sold_qty` from end-of-day `waste_qty`.
- Flattens the entire ledger into a **daily, product-level operational table**, serving as the pristine source of truth for all downstream modeling.

## 5. End-to-End Architecture Overview
The system is divided into strict, decoupled layers:
1. **Frontend UI:** A Next.js/React operational dashboard for bakery staff.
2. **Backend API:** A FastAPI layer serving inference requests via REST.
3. **Orchestrator:** A `LangGraph` DAG that manages the asynchronous execution of multiple models and business rules.
4. **Machine Learning Ensembles:** Dual-target forecasting using XGBoost (Primary) and PyTorch LSTMs (Challenger).
5. **Explanation Layer:** A constrained LLM proxy (`Google Gemini`) that translates mathematical tensors into plain-English staff briefings.

## 6. Operator Workflow
From the perspective of a bakery manager arriving at 5:00 AM:
1. The manager opens the **Frontend UI**.
2. They select a `Product` (e.g., Croissant), a `Target Date`, and the anticipated `Weather / Temperature` for the day.
3. They click **"Run Plan"**.
4. Behind the scenes, the system calculates the exact `sold_qty` forecast and the `waste_qty` buffer.
5. The UI displays the synthesized **Target Production Quantity**, alongside a conversational explanation of *why* the system recommends that number and any associated anomaly risks.

## 7. Frontend UI
**Built with Next.js & React (TypeScript, vanilla CSS)**
The frontend deliberately abandons developer-centric Swagger docs in favor of a polished, consumer-grade dashboard tailored for bakery operators.
- **Input Form:** Clean selectors for products, dates, and weather conditions.
- **Outputs:** Clear, color-coded metric cards prioritizing the final "Target Production" integer. 
- **Graceful States:** Includes smooth loading spinners, error boundary alerts, and visual "Anomaly Warnings" if the ensemble detects a structural break in the forecast logic.
- **Manager Diagnostics:** Technical payload arrays (like raw XGBoost weights) are hidden by default but accessible via a collapsible debug menu for system admins.

## 8. Backend API
**Built with FastAPI**
A minimal, lightning-fast web layer that translates JSON payloads into pandas DataFrames and triggers the LangGraph orchestrator.
- `GET /health`: Standard load-balancer ping.
- `POST /forecast`: The synchronous execution endpoint returning the compiled dual-target integer predictions and string explanations. 
*(Includes CORS middleware to natively support local React UI requests).*

## 9. Data Pipeline Lifecycle
1. **Raw CSV:** Ingestion of messy POS exports.
2. **Preprocessing:** Cleaned into `history_clean.csv`.
3. **Operational Table:** The source of truth for metrics.
4. **Feature Generation:** Expansion into wide matrices for ML consumption.

## 10. Feature Engineering
Found in `features.py`, the system generates dense, non-leaking predictive features:
- **Calendar:** Month, day-of-week, weekend flags, and cyclical trigonometric encodings.
- **Operational Lags:** 1, 2, 3, 7, 14, 28-day lags of sold/waste quantities.
- **Rolling Statistics:** 7-day, 14-day, 28-day moving averages and standard deviations.
- **Same-Weekday Priors:** Averaging the last 4 and 8 specific weekdays (e.g., the last 4 rainy Tuesdays).
- **Temperature Dynamics:** 1-day temperature deltas and categorical "Cold Shock" / "Heat Shock" flags.
- **Operational Ratios:** Fill-rates and Sell-through-rates.

## 11. Primary Models: XGBoost
**Dual-Target: `train_xgb.py`**
XGBoost serves as the primary production engine because it handles mixed tabular data (categorical weather, continuous temperatures) exceptionally well, trains instantly, and provides highly interpretable feature importance.
- Two distinct XGBoost models are trained per product: one explicitly predicting `sold_qty`, and the other predicting `waste_qty`.

## 12. Challenger Models: PyTorch LSTM
**Dual-Target: `train_lstm.py`**
A deep learning sequence model built as a structural challenger. 
- Learns global temporal patterns across *all* products simultaneously using categorical embeddings.
- Acts as a defensive fallback when recent sequences matter more than tabular snapshots.
- Also trained explicitly on the dual-target `sold` and `waste` pathways.

## 13. Deterministic Ensemble Logic
**Found in `ensemble.py`**
The system does not blindly average models. It applies distinct business rules:
- Compares the absolute gap between the XGBoost and LSTM predictions.
- **Rule:** If the gap is <= 5%, use a weighted blend.
- **Rule:** If the gap is massive (> 20%), flag an **Anomaly** and fall back heavily to the safer, smoother LSTM or historical moving averages.
- Target-specific gap thresholds ensure waste predictions are handled more conservatively than sales predictions.

## 14. Production Recommendation Logic
The final `recommended_made_qty` is not predicted directly by an AI. It is derived mathematically out of the dual-target forecasts:
`recommended_made_qty = predicted_sold_qty + predicted_waste_qty`
*This defines the total volume required on the shelf to capture the expected sales while absorbing the calculated structural waste buffer.*

## 15. LangGraph Inference Workflow
**Found in `forecast_graph.py`**
Online inference is managed by a Directed Acyclic Graph (DAG) state machine. It prevents monolithic function spaghetti by strictly bounding logic into concurrent nodes:
- `node_sold_pipeline` & `node_waste_pipeline` (run concurrently)
- `node_assemble_recommendation` (waits for both to finish)
- `node_explain` (translates the final assembled state to Gemini)

## 16. Gemini Explanation Layer
**Found in `explain.py`**
Large Language Models are incredibly powerful but notorious for hallucinating numbers. We utilize `google.generativeai` strictly under `Pydantic` schema constraints.
- **Rule:** The LLM is provided the *already-calculated, validated integers* and is instructed to generate a "staff-friendly explanation" of *why* those numbers make sense based on the weather and temperature context.
- The LLM **never** touches or alters the underlying forecast math.

## 17. Backtesting and Evaluation
**Found in `evaluate_backtest.py`**
The system abandons dangerous random Train/Test splits. All validation employs strict **Rolling-Origin Time-Based Splits** (e.g., train on January-October, test on November; train on January-November, test on December). It evaluates MAE and directional bias independently for the `sold` and `waste` targets.

## 18. Project Structure
```text
The_Baker_V3/
├── api.py                    # FastAPI server
├── ensemble.py               # Deterministic fusion rules
├── evaluate_backtest.py      # Validation and offline testing
├── explain.py                # Gemini LLM explanations
├── features.py               # Time-series feature engineering
├── forecast_graph.py         # LangGraph workflow orchestrator
├── preprocess_history.py     # Cleans raw logs into operational tables
├── train_lstm.py             # PyTorch Neural model training
├── train_xgb.py              # Primary gradient-boosted model training
│
├── frontend/                 # Next.js React Dashboard
│   ├── src/app/page.tsx      # Core UI layout and API fetch logic
│   └── src/app/globals.css   # Bakery-aesthetic UI styling
│
└── tests/
    └── test_smoke.py         # End-to-end integration boundaries
```

## 19. Running Locally
To launch the end-to-end ecosystem locally, you will need two terminal instances.

**Prerequisites:**
You must have a valid Gemini API key exposed in your environment:
```bash
export GEMINI_API_KEY="your_key_here"
```

**Terminal 1 (Backend API):**
```bash
cd The_Baker_V3
python api.py
# Server mounts at http://127.0.0.1:8000
```

**Terminal 2 (Frontend Dashboard):**
```bash
cd The_Baker_V3/frontend
npm i
npm run dev
# Dashboard mounts at http://localhost:3000
```

## 20. Deployment Notes
This architecture is container-ready. 
- The **FastAPI backend** should be packaged via Docker and deployed to a stateless runner (e.g., Google Cloud Run or AWS ECS). 
- The **Next.js frontend** can be statically compiled (`npm run build`) and natively hosted on Vercel or Netlify, pointing to the backend's production CNAME.
- All model artifacts (`.xgb`, `.pt`, `.pkl`) must be saved to a persistent blob storage bucket (S3/GCS) in production, rather than the local filesystem.

## 21. Screenshots
*(Placeholder: Insert screenshots of the Next.js `http://localhost:3000` Bakery Operations Dashboard showing the Input Form, Loading State, and Final Dual-Target Metric Cards here).*

## 22. Future Improvements
While operationally robust, the ecosystem has room to grow natively along the MLOps vector:
- **Model Tracking & Registry:** Migrate from locally saving `.xgb` and `.pt` files to a centralized experiment tracker such as **MLflow** or **Weights & Biases**, to formally monitor data-drift and pipeline decay.
- **Serverless Cloud Deployments:** Deploying the containerized FastAPI backend to scalable auto-managed infrastructure like **Google Cloud Run** or **AWS Elastic Container Service (ECS)** to dynamically autoscale inference endpoints during morning baker rushes.
- **Richer Operational Data:** Integrating promotions, local holidays, or marketing spend.
- **Better Demand Signals:** Parsing actual "Out of Stock" timestamps from the POS to calculate true unconstrained demand.
- **Improved Waste Diagnosis:** Distinguishing between "display waste" (needed for shelf aesthetics) vs "expiration waste".
- **Monitoring:** Integrating LangSmith or Datadog traces permanently into the `api.py` layer to monitor LLM explanation latency over time.
