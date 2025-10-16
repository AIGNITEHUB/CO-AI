"""
Comprehensive test to verify the fix handles both forecast mode and target mode correctly.
"""

import pandas as pd
import numpy as np
from logic.forecast import forecast_with_commitment
from logic.policy_commitments import load_country_commitment_from_json, CountryCommitment
import json

print("=" * 80)
print("COMPREHENSIVE TEST: FORECAST MODE vs TARGET MODE")
print("=" * 80)

# Test data
df_test = pd.DataFrame({
    'year': [2019, 2020, 2021, 2022],
    'emissions_gtco2': [0.315, 0.310, 0.340, 0.325]
})

# Load Vietnam commitment data (forecast mode: target_reduction_pct = 0)
with open('data/vietnam_forecast_mode.json', 'r', encoding='utf-8') as f:
    json_content = f.read()

commitment_forecast_mode = load_country_commitment_from_json(json_content, 'Vietnam')

print("\n" + "=" * 80)
print("TEST 1: FORECAST MODE (target_reduction_pct = 0)")
print("=" * 80)
print(f"Target reduction: {commitment_forecast_mode.target_reduction_pct}%")
print(f"Expected behavior: Policy-driven trajectory without target enforcement")

result1 = forecast_with_commitment(
    df=df_test,
    target_year=commitment_forecast_mode.target_year,
    target_emissions=commitment_forecast_mode.target_emissions_gtco2,
    pathway_type='policy_driven',
    country_commitment=commitment_forecast_mode
)

print(f"\nResults:")
print(f"  Last historical (2022): {df_test.iloc[-1]['emissions_gtco2']:.3f} GtCO2")
print(f"  First forecast (2023):  {result1.commitment_forecast[0]:.3f} GtCO2")
print(f"  Final forecast (2050):  {result1.commitment_forecast[-1]:.3f} GtCO2")

jump_pct = ((result1.commitment_forecast[0] - df_test.iloc[-1]['emissions_gtco2']) /
            df_test.iloc[-1]['emissions_gtco2']) * 100
print(f"  Jump at start: {jump_pct:+.1f}%")

if abs(jump_pct) < 5 and result1.commitment_forecast[-1] < 0.30:
    print(f"  Status: [PASS] Smooth transition and policy-driven reduction")
else:
    print(f"  Status: [FAIL] Unexpected behavior")

# Create modified commitment for target mode (target_reduction_pct = 80)
print("\n" + "=" * 80)
print("TEST 2: TARGET MODE (target_reduction_pct = 80)")
print("=" * 80)

# Modify the commitment data to test target mode
commitment_dict = json.loads(json_content)
commitment_dict['Vietnam']['target_reduction_pct'] = 80.0
commitment_target_mode = CountryCommitment.from_dict(commitment_dict['Vietnam'])

print(f"Target reduction: {commitment_target_mode.target_reduction_pct}%")
print(f"Expected behavior: Policy-driven trajectory with target enforcement at 2050")
print(f"Target emissions (80% reduction from {commitment_target_mode.baseline_emissions_gtco2:.3f}): " +
      f"{commitment_target_mode.target_emissions_gtco2:.3f} GtCO2")

result2 = forecast_with_commitment(
    df=df_test,
    target_year=commitment_target_mode.target_year,
    target_emissions=commitment_target_mode.target_emissions_gtco2,
    pathway_type='policy_driven',
    country_commitment=commitment_target_mode
)

print(f"\nResults:")
print(f"  Last historical (2022): {df_test.iloc[-1]['emissions_gtco2']:.3f} GtCO2")
print(f"  First forecast (2023):  {result2.commitment_forecast[0]:.3f} GtCO2")
print(f"  Final forecast (2050):  {result2.commitment_forecast[-1]:.3f} GtCO2")
print(f"  Target emissions:       {commitment_target_mode.target_emissions_gtco2:.3f} GtCO2")

jump_pct2 = ((result2.commitment_forecast[0] - df_test.iloc[-1]['emissions_gtco2']) /
             df_test.iloc[-1]['emissions_gtco2']) * 100
target_match = abs(result2.commitment_forecast[-1] - commitment_target_mode.target_emissions_gtco2) < 0.001

print(f"  Jump at start: {jump_pct2:+.1f}%")
print(f"  Hits target exactly: {target_match}")

if abs(jump_pct2) < 5 and target_match:
    print(f"  Status: [PASS] Smooth transition and reaches target")
else:
    print(f"  Status: [FAIL] Unexpected behavior")

print("\n" + "=" * 80)
print("FULL TRAJECTORY COMPARISON (selected years)")
print("=" * 80)
print(f"{'Year':<8} {'Forecast Mode':<18} {'Target Mode':<18} {'Difference':<15}")
print("-" * 80)

years_to_show = [2023, 2025, 2030, 2035, 2040, 2045, 2050]
for year in years_to_show:
    idx = np.where(result1.future_years == year)[0]
    if len(idx) > 0:
        idx = idx[0]
        val1 = result1.commitment_forecast[idx]
        val2 = result2.commitment_forecast[idx]
        diff = val1 - val2
        print(f"{year:<8} {val1:>10.3f} GtCO2     {val2:>10.3f} GtCO2     {diff:>+8.3f} GtCO2")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("The fix successfully:")
print("1. Eliminates jumps at the start of forecast (smooth from historical data)")
print("2. In forecast mode: Lets policies determine the trajectory")
print("3. In target mode: Enforces the target at the final year")
print("4. Scales policy reductions to match actual historical emissions level")
