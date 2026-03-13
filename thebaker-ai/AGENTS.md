# The Baker V3 - Agent Instructions

You are working on The Baker V3, an enterprise-grade retail demand forecasting system for a live bakery production platform.

## Product goal
Predict next-day product demand for bakery items using a hybrid deterministic system:
- Primary model: XGBoost for tabular forecasting
- Challenger model: PyTorch LSTM for sequence forecasting
- Ensemble logic: deterministic comparison and fallback rules
- Explanation layer: Gemini generates plain English staff explanations from validated numeric outputs only
- Orchestration layer: LangGraph workflow for online inference only, not for model training

## Hard rules
1. Never use an LLM to perform the numerical forecast itself.
2. Keep model training code separate from inference orchestration code.
3. Use rolling-origin time-based validation. Never use random train/test split.
4. XGBoost is the primary production model unless ensemble backtests clearly outperform it.
5. LangChain/LangGraph are only for online inference workflow and tool orchestration, not for model training.
6. Keep code modular, typed, and production-oriented.
7. Prefer plain Python modules over notebooks unless explicitly requested.
8. Add docstrings and brief comments for non-obvious logic.
9. When changing code, preserve existing working behavior unless the task explicitly asks for refactor/breaking changes.
10. If assumptions are needed, state them clearly in comments or in the final report.

## Current raw data schema
history.csv columns:
- date
- product
- qty
- weather
- temp

## Modeling assumptions
- Forecast unit = next-day quantity by product
- Data should be transformed into a daily product panel
- Missing dates must be handled explicitly
- Important engineered features:
  - calendar features
  - lag features
  - rolling statistics
  - same-weekday priors
  - temperature shock features
  - product-level prior features

## Expected modules
- features.py
- train_xgb.py
- train_lstm.py
- evaluate_backtest.py
- ensemble.py
- forecast_graph.py
- explain.py
- api.py
- tests/

## Done definition
A task is done only if:
1. The requested file(s) are implemented.
2. Imports are correct.
3. The code runs without obvious syntax errors.
4. Minimal tests or usage examples are included when appropriate.
5. A short summary is provided:
   - what changed
   - assumptions made
   - how to run it