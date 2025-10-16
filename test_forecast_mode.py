"""Test forecast mode when target_reduction_pct = 0"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from logic.policy_commitments import load_country_commitment

# Load Vietnam data with target_reduction_pct = 0
c = load_country_commitment('Vietnam', 'data')

print('=' * 70)
print('FORECAST MODE TEST: target_reduction_pct = 0')
print('=' * 70)
print()

print('📊 COMMITMENT PARAMETERS:')
print(f'  Baseline Year: {c.baseline_year}')
print(f'  Baseline Emissions: {c.baseline_emissions_gtco2} GtCO₂')
print(f'  Target Year: {c.target_year}')
print(f'  Target Reduction: {c.target_reduction_pct}% ⚠️ (0 = NO REDUCTION TARGET)')
print()

print('📈 SECTOR TARGETS:')
print('-' * 70)
for action in c.policy_actions:
    print(f'  {action.sector:12s}: Baseline={action.baseline_emissions_mtco2:6.1f} MtCO₂, '
          f'Target={action.target_pct:5.1f}%')
print()

print('🔮 FORECAST RESULTS (with target_reduction_pct = 0):')
print('-' * 70)
print(f"{'Year':<8} {'Reduction %':<15} {'Emissions (GtCO₂)':<20} {'Notes'}")
print('-' * 70)

test_years = [2025, 2026, 2028, 2030, 2035, 2040, 2045, 2050]

for year in test_years:
    if year < c.baseline_year:
        continue

    reduction_fraction = c.calculate_annual_reduction_fraction(year)
    reduction_pct = reduction_fraction * 100
    emissions = c.baseline_emissions_gtco2 * (1 - reduction_fraction)

    # Notes
    if year == c.baseline_year:
        note = "Baseline"
    elif year <= 2030:
        note = "Policy data"
    elif year == c.target_year:
        note = "Target year"
    else:
        note = "Extrapolated"

    print(f"{year:<8} {reduction_pct:>6.2f}%{'':<8} {emissions:>8.4f}{'':<12} {note}")

print()
print('=' * 70)
print('ANALYSIS')
print('=' * 70)
print()

# Check what happens with target = 0
reduction_2030 = c.calculate_annual_reduction_fraction(2030) * 100
reduction_2040 = c.calculate_annual_reduction_fraction(2040) * 100
reduction_2050 = c.calculate_annual_reduction_fraction(2050) * 100

print('📊 Reduction progression:')
print(f'  2030: {reduction_2030:.2f}% (last policy data)')
print(f'  2040: {reduction_2040:.2f}%')
print(f'  2050: {reduction_2050:.2f}% (target year)')
print()

if reduction_2050 > reduction_2030:
    print('⚠️ OBSERVATION: System still extrapolates toward sector targets')
    print('   Even though country target_reduction_pct = 0%')
    print('   Individual sectors have their own reduction_target_pct > 0')
    print()
    print('💡 INTERPRETATION:')
    print('   - Country target: 0% = No national commitment')
    print('   - Sector targets: Still active (Energy: 100%, Transport: 100%, etc.)')
    print('   - Result: Weighted sector reductions still applied')
elif reduction_2050 == reduction_2030:
    print('✅ System maintains 2030 levels (no further reduction)')
else:
    print('❌ Unexpected behavior')

print()
print('🔍 SECTOR-SPECIFIC EXTRAPOLATION:')
print('-' * 70)
for action in c.policy_actions:
    reduction_2030 = action.get_reduction_for_year(2030, c.target_year)
    reduction_2050 = action.get_reduction_for_year(2050, c.target_year)

    print(f'  {action.sector:12s}: 2030={reduction_2030:5.1f}% → 2050={reduction_2050:5.1f}% '
          f'(Target: {action.target_pct:5.1f}%)')

print()
print('=' * 70)
print('RECOMMENDATION FOR PURE FORECAST MODE')
print('=' * 70)
print()
print('If you want PURE FORECAST (no reduction target):')
print('  Option 1: Set all sector reduction_target_pct = 0')
print('  Option 2: Only use implementation_years data, no targets')
print('  Option 3: Implement separate "forecast_mode" flag')
print()
print('Current behavior:')
print('  ✅ Extrapolates based on SECTOR targets (not country target)')
print('  ✅ Each sector progresses toward its own goal')
print('  ⚠️  Country target_reduction_pct acts as validation/cap, not driver')
