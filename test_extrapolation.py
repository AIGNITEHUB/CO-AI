"""
Test auto-extrapolation when policy data has gaps.

Scenario: Data only extends to 2035, but target year is 2050.
Expected: Smooth linear extrapolation from 2035 → 2050.
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from logic.policy_commitments import CountryCommitment, PolicyAction

# Create Vietnam commitment with data only to 2035 (gap of 15 years)
commitment_with_gap = CountryCommitment(
    country="Vietnam",
    target_year=2050,
    target_reduction_pct=100.0,
    baseline_year=2020,
    baseline_emissions_gtco2=0.42,
    policy_actions=[
        PolicyAction(
            sector="Energy",
            action_name="Coal phase-out & Renewable expansion",
            baseline_year=2020,
            baseline_emissions_mtco2=210.0,
            reduction_target_pct=100.0,
            # Data only to 2035 (stopped at 60%)
            implementation_years=[2024, 2025, 2026, 2027, 2028, 2029, 2030, 2035],
            yearly_improvement_pct=[4.0, 8.5, 14.0, 20.0, 26.5, 33.0, 40.0, 60.0],
            status="On track"
        ),
        PolicyAction(
            sector="Transport",
            action_name="Electric vehicle adoption",
            baseline_year=2020,
            baseline_emissions_mtco2=85.0,
            reduction_target_pct=100.0,
            # Data only to 2035 (stopped at 55%)
            implementation_years=[2024, 2025, 2026, 2027, 2028, 2029, 2030, 2035],
            yearly_improvement_pct=[2.0, 5.0, 9.0, 14.0, 20.0, 26.0, 32.0, 55.0],
            status="Behind schedule"
        ),
        PolicyAction(
            sector="Industry",
            action_name="Energy efficiency in manufacturing",
            baseline_year=2020,
            baseline_emissions_mtco2=75.0,
            reduction_target_pct=100.0,
            # Data only to 2035 (stopped at 52%)
            implementation_years=[2024, 2025, 2026, 2027, 2028, 2029, 2030, 2035],
            yearly_improvement_pct=[3.0, 7.0, 11.5, 16.0, 20.5, 25.0, 30.0, 52.0],
            status="On track"
        ),
        PolicyAction(
            sector="Agriculture",
            action_name="Sustainable rice production",
            baseline_year=2020,
            baseline_emissions_mtco2=35.0,
            reduction_target_pct=90.0,
            # Data only to 2035 (stopped at 45%)
            implementation_years=[2024, 2025, 2026, 2027, 2028, 2029, 2030, 2035],
            yearly_improvement_pct=[2.5, 6.0, 10.0, 14.0, 18.0, 22.0, 28.0, 45.0],
            status="Ahead"
        ),
        PolicyAction(
            sector="Forestry",
            action_name="Reforestation & Forest conservation",
            baseline_year=2020,
            baseline_emissions_mtco2=-15.0,
            removal_increase_pct=200.0,
            # Data only to 2035 (stopped at 100%)
            implementation_years=[2024, 2025, 2026, 2027, 2028, 2029, 2030, 2035],
            yearly_improvement_pct=[8.0, 16.0, 25.0, 34.0, 43.0, 52.0, 62.0, 100.0],
            status="Ahead"
        )
    ]
)

# Link parent to children for auto-calculation of share_pct
for action in commitment_with_gap.policy_actions:
    action._parent_commitment = commitment_with_gap

print("=" * 80)
print("TEST: Auto-Extrapolation with Data Gap (2035 → 2050)")
print("=" * 80)
print()

# Test years across the gap
test_years = [2024, 2030, 2035, 2037, 2040, 2043, 2045, 2048, 2050]

print("Testing reduction fraction calculation with auto-extrapolation:\n")
print(f"{'Year':<8} {'Reduction %':<15} {'Emissions (GtCO2)':<20} {'Notes'}")
print("-" * 80)

baseline = commitment_with_gap.baseline_emissions_gtco2

for year in test_years:
    reduction_fraction = commitment_with_gap.calculate_annual_reduction_fraction(year)
    reduction_pct = reduction_fraction * 100
    emissions = baseline * (1 - reduction_fraction)

    # Determine notes
    if year <= 2035:
        note = "Data point"
    elif year < 2050:
        note = "EXTRAPOLATED"
    else:
        note = "Target year"

    print(f"{year:<8} {reduction_pct:>6.2f}%{'':<8} {emissions:>8.4f}{'':<12} {note}")

print()
print("=" * 80)
print("VALIDATION CHECKS")
print("=" * 80)
print()

# Check 1: No plateau effect
reduction_2035 = commitment_with_gap.calculate_annual_reduction_fraction(2035) * 100
reduction_2040 = commitment_with_gap.calculate_annual_reduction_fraction(2040) * 100
reduction_2045 = commitment_with_gap.calculate_annual_reduction_fraction(2045) * 100
reduction_2050 = commitment_with_gap.calculate_annual_reduction_fraction(2050) * 100

print("1. Check for plateau effect (should progressively increase):")
print(f"   2035: {reduction_2035:.2f}%")
print(f"   2040: {reduction_2040:.2f}% (Δ = +{reduction_2040 - reduction_2035:.2f}%)")
print(f"   2045: {reduction_2045:.2f}% (Δ = +{reduction_2045 - reduction_2040:.2f}%)")
print(f"   2050: {reduction_2050:.2f}% (Δ = +{reduction_2050 - reduction_2045:.2f}%)")

plateau_check = reduction_2040 > reduction_2035 and reduction_2045 > reduction_2040
print(f"   ✓ PASS: Continuous progression" if plateau_check else f"   ✗ FAIL: Plateau detected")
print()

# Check 2: Reaches target by 2050
target_reached = abs(reduction_2050 - commitment_with_gap.target_reduction_pct) < 0.1
print(f"2. Check target reached by 2050:")
print(f"   Target: {commitment_with_gap.target_reduction_pct:.2f}%")
print(f"   Actual: {reduction_2050:.2f}%")
print(f"   ✓ PASS: Target reached" if target_reached else f"   ✗ FAIL: Target not reached")
print()

# Check 3: Extrapolation is linear (approximately)
# Check that the rate of increase is roughly constant
rate_2035_2040 = (reduction_2040 - reduction_2035) / 5
rate_2040_2045 = (reduction_2045 - reduction_2040) / 5
rate_2045_2050 = (reduction_2050 - reduction_2045) / 5

print("3. Check extrapolation linearity (rate should be roughly constant):")
print(f"   2035-2040: {rate_2035_2040:.3f}% per year")
print(f"   2040-2045: {rate_2040_2045:.3f}% per year")
print(f"   2045-2050: {rate_2045_2050:.3f}% per year")

# Allow some variation due to rounding and weighted calculation
rate_variation = max(rate_2035_2040, rate_2040_2045, rate_2045_2050) - min(rate_2035_2040, rate_2040_2045, rate_2045_2050)
linear_check = rate_variation < 0.5  # Less than 0.5% variation

print(f"   Max variation: {rate_variation:.3f}%")
print(f"   ✓ PASS: Roughly linear" if linear_check else f"   ✗ FAIL: Non-linear")
print()

# Check 4: Sector-specific extrapolation
print("4. Check sector-specific extrapolation:")
for action in commitment_with_gap.policy_actions:
    last_year = action.implementation_years[-1]
    last_reduction = action.yearly_improvement_pct[-1]

    # Get extrapolated value at 2040
    extrapolated_2040 = action.get_reduction_for_year(2040, target_year=commitment_with_gap.target_year)

    # Calculate expected slope
    years_to_target = commitment_with_gap.target_year - last_year
    slope = (action.target_pct - last_reduction) / years_to_target
    expected_2040 = last_reduction + slope * (2040 - last_year)

    match = abs(extrapolated_2040 - expected_2040) < 0.01

    print(f"   {action.sector:12s}: Last={last_reduction:5.1f}% → 2040={extrapolated_2040:5.1f}% (expected {expected_2040:5.1f}%)", end="")
    print(" ✓" if match else f" ✗ (diff: {abs(extrapolated_2040 - expected_2040):.2f}%)")

print()
print("=" * 80)
print("COMPARISON WITH EXTENDED DATA")
print("=" * 80)
print()

# Load full Vietnam data with extended milestones
from logic.policy_commitments import load_country_commitment

try:
    commitment_extended = load_country_commitment("Vietnam", data_dir="data")

    print("Comparing extrapolated vs actual extended data:\n")
    print(f"{'Year':<8} {'Extrapolated':<15} {'Extended Data':<15} {'Difference':<15} {'% Diff'}")
    print("-" * 80)

    comparison_years = [2035, 2040, 2045, 2050]
    for year in comparison_years:
        reduction_gap = commitment_with_gap.calculate_annual_reduction_fraction(year) * 100
        reduction_ext = commitment_extended.calculate_annual_reduction_fraction(year) * 100

        diff = reduction_gap - reduction_ext
        pct_diff = (diff / reduction_ext * 100) if reduction_ext != 0 else 0

        print(f"{year:<8} {reduction_gap:>6.2f}%{'':<8} {reduction_ext:>6.2f}%{'':<8} {diff:>+6.2f}%{'':<8} {pct_diff:>+6.1f}%")

    print()
    print("Notes:")
    print("- Small differences are expected because extended data has non-linear milestones")
    print("- Extrapolation uses linear progression, while extended data may have acceleration")
    print("- Both should reach similar values by 2050 (target year)")

except Exception as e:
    print(f"Could not load extended data: {e}")
    print("Comparison skipped.")

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
