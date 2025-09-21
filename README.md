# CO₂ AI Demo (Basic)

This Streamlit demo provides **two functions**:

1. **Filter & Forecast Global CO₂ Emissions**
   - Load the provided sample dataset (`data/sample_co2_global.csv`) or upload your own CSV with columns: `year`, `emissions_gtco2`.
   - Filter by year, visualize the time series, and **forecast 1–30 years** ahead using a simple polynomial regression (sklearn).

2. **CO₂ → Biofuels Conversion Predictor**
   - Enter pathway, catalyst family, and operating conditions (T/P/H₂:CO₂).
   - The demo returns a **predicted primary product** and a **heuristic yield/efficiency** (for teaching only).
   - You can export the results to CSV.

> ⚠️ This is an educational demo. Numbers are **not** real experimental predictions.
> Replace the dataset and logic with your own models/data for real work.

## Quick start

```bash
# 1) Create and activate virtual env (example with Python 3.10+)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

The app opens in your browser (usually http://localhost:8501).

## Files

- `app.py` — Streamlit UI with two tabs.
- `logic/forecast.py` — Emissions forecasting helpers.
- `logic/conversion.py` — CO₂→biofuels predictor logic (heuristic + tiny ML stub).
- `data/sample_co2_global.csv` — Toy dataset for demo.
- `requirements.txt`, `README.md`

## Notes

- Charts use matplotlib (no seaborn).
- Forecasting uses a simple polynomial Regression (degree configurable). Replace with ARIMA/Prophet/etc. if needed.
- Conversion predictor combines a rule-based baseline with a small synthetic-ML regressor for illustrative purposes.
