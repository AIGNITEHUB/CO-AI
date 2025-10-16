"""
Test script for Vietnam policy commitment tracking.
"""

import sys
import numpy as np
import pandas as pd

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from logic.policy_commitments import load_country_commitment, get_available_countries
from logic.forecast import forecast_with_commitment

# Test 1: Load Vietnam commitment data
print("=" * 60)
print("Test 1: Loading Vietnam commitment data")
print("=" * 60)

try:
    available = get_available_countries()
    print(f"Available countries: {available}")

    vietnam = load_country_commitment("Vietnam")
    print(f"\n✓ Successfully loaded {vietnam.country} commitment data")
    print(f"  Target Year: {vietnam.target_year}")
    print(f"  Target Reduction: {vietnam.target_reduction_pct}%")
    print(f"  Baseline Year: {vietnam.baseline_year}")
    print(f"  Baseline Emissions: {vietnam.baseline_emissions_gtco2} GtCO₂")
    print(f"  Target Emissions: {vietnam.target_emissions_gtco2:.3f} GtCO₂")
    print(f"  Number of Policy Actions: {len(vietnam.policy_actions)}")

    print("\n  Policy Actions:")
    for action in vietnam.policy_actions:
        print(f"    - {action.sector}: {action.action_name}")
        print(f"      Baseline: {action.baseline_emissions_mtco2} MtCO₂")
        print(f"      Share: {action.share_pct}%")
        print(f"      Target Reduction: {action.reduction_target_pct}%")
        print(f"      Status: {action.status}")

except Exception as e:
    print(f"✗ Error loading Vietnam data: {e}")
    exit(1)

# Test 2: Calculate sector breakdown
print("\n" + "=" * 60)
print("Test 2: Sector breakdown")
print("=" * 60)

try:
    sector_breakdown = vietnam.get_sector_breakdown()
    print("\nSector emissions (baseline):")
    for sector, emissions in sector_breakdown.items():
        print(f"  {sector}: {emissions:.1f} MtCO₂")

    sector_contributions = vietnam.get_sector_contributions()
    print("\nSector reduction contributions:")
    for sector, reduction in sector_contributions.items():
        print(f"  {sector}: {reduction:.1f} MtCO₂")

    print("\n✓ Sector calculations successful")

except Exception as e:
    print(f"✗ Error calculating sectors: {e}")
    exit(1)

# Test 3: Test annual reduction calculation
print("\n" + "=" * 60)
print("Test 3: Annual reduction calculation")
print("=" * 60)

try:
    test_years = [2024, 2025, 2027, 2030]
    print("\nReduction fractions by year:")
    for year in test_years:
        reduction = vietnam.calculate_annual_reduction_fraction(year)
        print(f"  {year}: {reduction:.4f} ({reduction*100:.2f}%)")

    print("\n✓ Annual reduction calculation successful")

except Exception as e:
    print(f"✗ Error calculating annual reductions: {e}")
    exit(1)

# Test 4: Test policy-driven pathway with sample data
print("\n" + "=" * 60)
print("Test 4: Policy-driven pathway forecast")
print("=" * 60)

try:
    # Create sample historical data for Vietnam
    years = np.arange(2000, 2024)
    # Realistic growth trajectory for Vietnam
    emissions = 0.15 + 0.011 * (years - 2000) + 0.0002 * (years - 2000)**2
    emissions = np.clip(emissions, 0, None)

    sample_df = pd.DataFrame({
        'year': years,
        'emissions_gtco2': emissions
    })

    print(f"\nSample data: {len(sample_df)} years ({sample_df['year'].min()}-{sample_df['year'].max()})")
    print(f"Latest emissions: {sample_df.iloc[-1]['emissions_gtco2']:.3f} GtCO₂")

    # Generate policy-driven forecast using target from commitment data
    target_emissions = vietnam.target_emissions_gtco2

    result = forecast_with_commitment(
        df=sample_df,
        target_year=vietnam.target_year,
        target_emissions=target_emissions,
        pathway_type='policy_driven',
        country_commitment=vietnam,
        bau_degree=2
    )

    print(f"\n✓ Policy-driven forecast generated successfully")
    print(f"  Pathway type: {result.pathway_type}")
    print(f"  BAU model: {result.bau_model_name}")
    print(f"  Future years: {len(result.future_years)} years")
    print(f"  Current emissions: {result.current_emissions:.3f} GtCO₂")
    print(f"  Target emissions: {result.target_emissions:.3f} GtCO₂")

    # Show sample projections
    print("\n  Sample projections (every 5 years):")
    for i in range(0, len(result.future_years), 5):
        year = result.future_years[i]
        bau = result.bau_forecast[i]
        policy = result.commitment_forecast[i]
        gap = result.emissions_gap[i]
        print(f"    {int(year)}: BAU={bau:.3f}, Policy={policy:.3f}, Gap={gap:.3f} GtCO₂")

    # Verify pathway makes sense
    assert result.commitment_forecast[0] < result.bau_forecast[0], "Policy pathway should be lower than BAU"
    assert result.commitment_forecast[-1] <= result.target_emissions + 0.01, "Should reach target"
    assert all(result.commitment_forecast[i] >= result.commitment_forecast[i+1] for i in range(len(result.commitment_forecast)-1)), "Should be decreasing"

    print("\n✓ All validation checks passed")

except Exception as e:
    print(f"✗ Error in policy-driven forecast: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Compare with exponential pathway
print("\n" + "=" * 60)
print("Test 5: Compare policy-driven vs exponential pathway")
print("=" * 60)

try:
    # Generate exponential pathway for comparison
    result_exp = forecast_with_commitment(
        df=sample_df,
        target_year=vietnam.target_year,
        target_emissions=target_emissions,
        pathway_type='exponential',
        bau_degree=2
    )

    print("\nComparison at key years:")
    print(f"{'Year':<6} {'Policy-Driven':<15} {'Exponential':<15} {'Difference':<10}")
    print("-" * 50)

    for i in [0, 5, 10, 15, 20, 25]:
        if i < len(result.future_years):
            year = int(result.future_years[i])
            policy_val = result.commitment_forecast[i]
            exp_val = result_exp.commitment_forecast[i]
            diff = policy_val - exp_val

            print(f"{year:<6} {policy_val:>8.3f} GtCO₂  {exp_val:>8.3f} GtCO₂  {diff:>+8.3f}")

    print("\n✓ Pathway comparison successful")

except Exception as e:
    print(f"✗ Error in pathway comparison: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED")
print("=" * 60)
print("\nVietnam policy commitment tracking is ready to use!")
print("Run 'streamlit run app.py' and select Vietnam to see it in action.")
