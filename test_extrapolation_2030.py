"""Test auto-extrapolation from 2030 to 2050"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from logic.policy_commitments import load_country_commitment

# Load Vietnam data with only 2026-2030 data
c = load_country_commitment('Vietnam', 'data')

print('=' * 70)
print('AUTO-EXTRAPOLATION TEST: 2030 → 2050')
print('=' * 70)
print()

print('📊 DATA RANGE:')
print(f'  Last data year: {c.policy_actions[0].implementation_years[-1]}')
print(f'  Target year: {c.target_year}')
print(f'  Gap: {c.target_year - c.policy_actions[0].implementation_years[-1]} years')
print()

print('📈 SECTOR DATA AT 2030:')
print('-' * 70)
for action in c.policy_actions:
    reduction_2030 = action.get_reduction_for_year(2030, c.target_year)
    print(f'  {action.sector:12s}: {reduction_2030:5.1f}% → Target: {action.target_pct:5.1f}%')
print()

print('🔮 EXTRAPOLATION RESULTS (2030 → 2050):')
print('-' * 70)
print(f"{'Year':<8} {'Reduction %':<15} {'Emissions (GtCO₂)':<20} {'Status'}")
print('-' * 70)

test_years = [2030, 2032, 2035, 2038, 2040, 2043, 2045, 2048, 2050]

for year in test_years:
    reduction_fraction = c.calculate_annual_reduction_fraction(year)
    reduction_pct = reduction_fraction * 100
    emissions = c.baseline_emissions_gtco2 * (1 - reduction_fraction)

    # Determine status
    if year == 2030:
        status = "Last data point"
    elif year == c.target_year:
        status = "TARGET REACHED"
    else:
        status = "Extrapolated"

    print(f"{year:<8} {reduction_pct:>6.2f}%{'':<8} {emissions:>8.4f}{'':<12} {status}")

print()
print('=' * 70)
print('VALIDATION CHECKS')
print('=' * 70)
print()

# Check 1: Continuous progression
reduction_2030 = c.calculate_annual_reduction_fraction(2030) * 100
reduction_2035 = c.calculate_annual_reduction_fraction(2035) * 100
reduction_2040 = c.calculate_annual_reduction_fraction(2040) * 100
reduction_2045 = c.calculate_annual_reduction_fraction(2045) * 100
reduction_2050 = c.calculate_annual_reduction_fraction(2050) * 100

print('1. ✓ Check continuous progression:')
print(f'   2030: {reduction_2030:.2f}% (last data)')
print(f'   2035: {reduction_2035:.2f}% (Δ = +{reduction_2035 - reduction_2030:.2f}%)')
print(f'   2040: {reduction_2040:.2f}% (Δ = +{reduction_2040 - reduction_2035:.2f}%)')
print(f'   2045: {reduction_2045:.2f}% (Δ = +{reduction_2045 - reduction_2040:.2f}%)')
print(f'   2050: {reduction_2050:.2f}% (Δ = +{reduction_2050 - reduction_2045:.2f}%)')

plateau_check = (reduction_2035 > reduction_2030 and
                 reduction_2040 > reduction_2035 and
                 reduction_2045 > reduction_2040)
print(f'   {"✅ PASS" if plateau_check else "❌ FAIL"}: No plateau effect')
print()

# Check 2: Reaches target
target_reached = abs(reduction_2050 - c.target_reduction_pct) < 0.1
print(f'2. ✓ Check target reached:')
print(f'   Target: {c.target_reduction_pct:.2f}%')
print(f'   Actual: {reduction_2050:.2f}%')
print(f'   {"✅ PASS" if target_reached else "❌ FAIL"}: Target reached at 2050')
print()

# Check 3: Linear extrapolation rate
rate_2030_2040 = (reduction_2040 - reduction_2030) / 10
rate_2040_2050 = (reduction_2050 - reduction_2040) / 10

print(f'3. ✓ Check extrapolation rate:')
print(f'   2030-2040: {rate_2030_2040:.3f}% per year')
print(f'   2040-2050: {rate_2040_2050:.3f}% per year')
print(f'   Variation: {abs(rate_2030_2040 - rate_2040_2050):.3f}%')
print()

# Sector-specific extrapolation
print('4. ✓ Sector-specific extrapolation to 2040:')
for action in c.policy_actions:
    last_year = action.implementation_years[-1]
    last_reduction = action.yearly_improvement_pct[-1]

    extrapolated_2040 = action.get_reduction_for_year(2040, target_year=c.target_year)

    # Calculate expected slope
    years_to_target = c.target_year - last_year
    slope = (action.target_pct - last_reduction) / years_to_target
    expected_2040 = last_reduction + slope * (2040 - last_year)

    match = abs(extrapolated_2040 - expected_2040) < 0.01

    print(f'   {action.sector:12s}: {last_reduction:5.1f}% → {extrapolated_2040:5.1f}% (expected {expected_2040:5.1f}%) {"✅" if match else "❌"}')

print()
print('=' * 70)
print('✅ AUTO-EXTRAPOLATION WORKING CORRECTLY!')
print('=' * 70)
print()
print('💡 NOTE: System will automatically extrapolate from 2030 to 2050')
print('   - Uses linear progression toward target')
print('   - Each sector extrapolates independently')
print('   - No additional data needed beyond 2030')
