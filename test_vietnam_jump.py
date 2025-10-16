"""
Test to reproduce the Vietnam 2050 jump issue.
Expected: Smooth transition from historical data to forecast.
Actual: Jump from 0.337 to 0.45 when using forecast mode.
"""

import pandas as pd
import numpy as np
from logic.forecast import forecast_with_commitment
from logic.policy_commitments import load_country_commitment_from_json
import json

# Create test historical data ending at 2022 (where issue is visible)
df_test = pd.DataFrame({
    'year': [2019, 2020, 2021, 2022],
    'emissions_gtco2': [0.315, 0.310, 0.340, 0.325]
})

# Load Vietnam commitment data
with open('data/vietnam_forecast_mode.json', 'r', encoding='utf-8') as f:
    json_content = f.read()

commitment_data = load_country_commitment_from_json(json_content, 'Vietnam')

print("=" * 70)
print("TESTING VIETNAM FORECAST MODE JUMP ISSUE")
print("=" * 70)
print(f"\nHistorical data (last 4 years):")
print(df_test)
print(f"\nLast historical emissions: {df_test.iloc[-1]['emissions_gtco2']:.3f} GtCO2 (2022)")
print(f"Baseline from JSON: {commitment_data.baseline_emissions_gtco2:.3f} GtCO2 (2025)")
print(f"Target year: {commitment_data.target_year}")
print(f"Target reduction: {commitment_data.target_reduction_pct}%")

# Run forecast
result = forecast_with_commitment(
    df=df_test,
    target_year=commitment_data.target_year,
    target_emissions=commitment_data.target_emissions_gtco2,
    pathway_type='policy_driven',
    country_commitment=commitment_data
)

print(f"\n{'='*70}")
print("FORECAST RESULTS (First 5 years)")
print(f"{'='*70}")
print(f"{'Year':<8} {'Commitment':<15} {'Change from prev':<20}")
print("-" * 70)

prev_value = df_test.iloc[-1]['emissions_gtco2']
for i in range(min(5, len(result.future_years))):
    year = result.future_years[i]
    commit = result.commitment_forecast[i]
    change = commit - prev_value
    change_pct = (change / prev_value) * 100

    print(f"{int(year):<8} {commit:>8.3f} GtCO2   {change:>+8.3f} ({change_pct:>+6.1f}%)")
    prev_value = commit

# Check for jump in first year
first_forecast = result.commitment_forecast[0]
last_historical = df_test.iloc[-1]['emissions_gtco2']
jump = first_forecast - last_historical
jump_pct = (jump / last_historical) * 100

print(f"\n{'='*70}")
print("ISSUE DETECTED:" if abs(jump_pct) > 10 else "NO MAJOR ISSUE")
print(f"{'='*70}")
print(f"Jump from 2022 to 2023: {jump:+.3f} GtCO2 ({jump_pct:+.1f}%)")

if abs(jump_pct) > 10:
    print(f"\n[WARNING] PROBLEM: Large discontinuity detected!")
    print(f"   Expected: Smooth transition from historical data")
    print(f"   Actual: Jump of {abs(jump_pct):.1f}% in first forecast year")
    print(f"\n   Root cause: Using baseline ({commitment_data.baseline_emissions_gtco2:.3f}) instead of")
    print(f"               last historical value ({last_historical:.3f}) as starting point")
else:
    print(f"[OK] Smooth transition from historical to forecast")

# Also check 2050
idx_2050 = np.where(result.future_years == 2050)[0]
if len(idx_2050) > 0:
    value_2050 = result.commitment_forecast[idx_2050[0]]
    print(f"\n2050 forecast: {value_2050:.3f} GtCO2")
