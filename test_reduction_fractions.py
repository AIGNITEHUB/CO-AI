"""
Test to check what reduction fractions are being calculated over time.
This will help us understand why the 2050 value jumps to 0.45.
"""

import json
from logic.policy_commitments import load_country_commitment_from_json

# Load Vietnam commitment data
with open('data/vietnam_forecast_mode.json', 'r', encoding='utf-8') as f:
    json_content = f.read()

commitment_data = load_country_commitment_from_json(json_content, 'Vietnam')

print("=" * 80)
print("REDUCTION FRACTION ANALYSIS FOR VIETNAM")
print("=" * 80)
print(f"\nBaseline emissions: {commitment_data.baseline_emissions_gtco2:.3f} GtCO2")
print(f"Target year: {commitment_data.target_year}")
print(f"Target reduction: {commitment_data.target_reduction_pct}%")

# Check policy data range
max_data_year = max(
    max(action.implementation_years) if action.implementation_years else 0
    for action in commitment_data.policy_actions
)
print(f"Policy data extends to: {max_data_year}")

print(f"\n{'='*80}")
print(f"{'Year':<8} {'Reduction Fraction':<20} {'Calculated Emissions':<20}")
print(f"{'='*80}")

# Test years: early years (with data), middle years (extrapolation), and 2050
test_years = [2023, 2025, 2026, 2030, 2035, 2040, 2045, 2050]

for year in test_years:
    reduction_fraction = commitment_data.calculate_annual_reduction_fraction(year)
    calculated_emissions = commitment_data.baseline_emissions_gtco2 * (1 - reduction_fraction)

    marker = ""
    if year <= max_data_year:
        marker = " (has data)"
    elif year == 2050:
        marker = " (TARGET YEAR)"
    else:
        marker = " (extrapolated)"

    print(f"{year:<8} {reduction_fraction:<20.4f} {calculated_emissions:<20.3f} {marker}")

print(f"\n{'='*80}")
print("DETAILED SECTOR ANALYSIS FOR KEY YEARS")
print(f"{'='*80}")

for year in [2030, 2040, 2050]:
    print(f"\n--- Year {year} ---")
    for action in commitment_data.policy_actions:
        reduction_pct = action.get_reduction_for_year(year, target_year=commitment_data.target_year)
        print(f"  {action.sector:<15} {reduction_pct:>6.2f}% (baseline: {action.baseline_emissions_mtco2:>6.1f} MtCO2)")

print(f"\n{'='*80}")
print("ANALYSIS")
print(f"{'='*80}")
print(f"\nIf reduction_fraction approaches 0 over time, calculated_emissions")
print(f"will return to baseline_emissions ({commitment_data.baseline_emissions_gtco2:.3f})")
print(f"\nThis causes the jump from ~0.32 to 0.45 in 2050!")
