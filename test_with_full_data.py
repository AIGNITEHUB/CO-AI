"""
Test with the actual full Vietnam dataset that includes 2025 data.
"""

import pandas as pd
from logic.forecast import forecast_with_commitment
from logic.policy_commitments import load_country_commitment_from_json
import json

# Load actual Vietnam historical data
df_vietnam = pd.read_csv('data/vietnam_co2_emissions.csv')

print("=" * 80)
print("TESTING WITH FULL VIETNAM HISTORICAL DATA")
print("=" * 80)
print(f"\nHistorical data range: {df_vietnam['year'].min()} - {df_vietnam['year'].max()}")
print(f"Last few years:")
print(df_vietnam.tail(8).to_string(index=False))

# Load Vietnam commitment data
with open('data/vietnam_forecast_mode.json', 'r', encoding='utf-8') as f:
    json_content = f.read()

commitment_data = load_country_commitment_from_json(json_content, 'Vietnam')

print(f"\n{'='*80}")
print("COMMITMENT DATA")
print(f"{'='*80}")
print(f"Baseline year: {commitment_data.baseline_year}")
print(f"Baseline emissions: {commitment_data.baseline_emissions_gtco2:.3f} GtCO2")
print(f"Target year: {commitment_data.target_year}")
print(f"Target reduction: {commitment_data.target_reduction_pct}%")
print(f"Target emissions: {commitment_data.target_emissions_gtco2:.3f} GtCO2")

last_historical = df_vietnam.iloc[-1]['emissions_gtco2']
print(f"\nLast historical emissions (2025): {last_historical:.3f} GtCO2")
print(f"Baseline emissions (2025): {commitment_data.baseline_emissions_gtco2:.3f} GtCO2")
print(f"Match: {last_historical == commitment_data.baseline_emissions_gtco2}")

# Run forecast
print(f"\n{'='*80}")
print("RUNNING FORECAST")
print(f"{'='*80}")

result = forecast_with_commitment(
    df=df_vietnam,
    target_year=commitment_data.target_year,
    target_emissions=commitment_data.target_emissions_gtco2,
    pathway_type='policy_driven',
    country_commitment=commitment_data
)

print(f"\nForecast results:")
print(f"{'Year':<8} {'Commitment Forecast':<20} {'BAU Forecast':<20}")
print("-" * 80)

# Show key years
key_years = [2026, 2030, 2035, 2040, 2045, 2050]
for year in key_years:
    idx = result.future_years == year
    if idx.any():
        idx_pos = idx.argmax()
        commit = result.commitment_forecast[idx_pos]
        bau = result.bau_forecast[idx_pos]
        print(f"{year:<8} {commit:>12.3f} GtCO2     {bau:>12.3f} GtCO2")

print(f"\n{'='*80}")
print("ISSUE CHECK")
print(f"{'='*80}")

value_2050 = result.commitment_forecast[-1]
print(f"2050 forecast value: {value_2050:.3f} GtCO2")
print(f"Expected (based on policies): ~0.147 GtCO2")
print(f"Target from JSON: {commitment_data.target_emissions_gtco2:.3f} GtCO2")

if abs(value_2050 - 0.45) < 0.01:
    print(f"\n[PROBLEM] 2050 value is stuck at baseline (0.45)")
elif abs(value_2050 - 0.50) < 0.01:
    print(f"\n[PROBLEM] 2050 value jumped to 0.50")
elif value_2050 < 0.20:
    print(f"\n[OK] 2050 value shows policy-driven reduction")
else:
    print(f"\n[CHECK] 2050 value is {value_2050:.3f}")
