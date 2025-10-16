"""Test PURE FORECAST MODE - all targets set to 0"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from logic.policy_commitments import load_country_commitment_from_json
import json

# Load Vietnam data with ALL targets = 0
with open('data/vietnam_forecast_mode.json', 'r', encoding='utf-8') as f:
    json_content = f.read()

c = load_country_commitment_from_json(json_content, 'Vietnam')

print('=' * 70)
print('PURE FORECAST MODE TEST')
print('All sector targets = 0 (no reduction goals)')
print('=' * 70)
print()

print('📊 CONFIGURATION:')
print(f'  Country target: {c.target_reduction_pct}%')
print(f'  Baseline: {c.baseline_emissions_gtco2} GtCO₂ ({c.baseline_year})')
print(f'  Data range: {c.policy_actions[0].implementation_years[0]}-{c.policy_actions[0].implementation_years[-1]}')
print()

print('📈 SECTOR CONFIGURATIONS:')
print('-' * 70)
for action in c.policy_actions:
    print(f'  {action.sector:12s}: Target={action.target_pct}%, Last data (2030)={action.yearly_improvement_pct[-1]}%')
print()

print('🔮 FORECAST RESULTS (Pure Forecast - No Targets):')
print('-' * 70)
print(f"{'Year':<8} {'Reduction %':<15} {'Emissions (GtCO₂)':<20} {'Notes'}")
print('-' * 70)

test_years = [2025, 2026, 2028, 2030, 2032, 2035, 2040, 2045, 2050]

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
        note = "Target year (no target set)"
    else:
        note = "Forecast (maintained at 2030 level)"

    print(f"{year:<8} {reduction_pct:>6.2f}%{'':<8} {emissions:>8.4f}{'':<12} {note}")

print()
print('=' * 70)
print('VALIDATION')
print('=' * 70)
print()

reduction_2030 = c.calculate_annual_reduction_fraction(2030) * 100
reduction_2035 = c.calculate_annual_reduction_fraction(2035) * 100
reduction_2040 = c.calculate_annual_reduction_fraction(2040) * 100
reduction_2050 = c.calculate_annual_reduction_fraction(2050) * 100

emissions_2030 = c.baseline_emissions_gtco2 * (1 - reduction_2030/100)
emissions_2040 = c.baseline_emissions_gtco2 * (1 - reduction_2040/100)
emissions_2050 = c.baseline_emissions_gtco2 * (1 - reduction_2050/100)

print('📊 Key metrics:')
print(f'  2030 (last data): {reduction_2030:.2f}% reduction → {emissions_2030:.3f} GtCO₂')
print(f'  2040 (forecast):  {reduction_2040:.2f}% reduction → {emissions_2040:.3f} GtCO₂')
print(f'  2050 (forecast):  {reduction_2050:.2f}% reduction → {emissions_2050:.3f} GtCO₂')
print()

if abs(reduction_2030 - reduction_2040) < 0.1 and abs(reduction_2030 - reduction_2050) < 0.1:
    print('✅ SUCCESS: Pure forecast mode working correctly!')
    print('   System maintains 2030 emission levels (no further reduction)')
    print(f'   Stable at ~{reduction_2030:.2f}% reduction from baseline')
    print()
    print('📈 Interpretation:')
    print('   - 2026-2030: Policy implementation reduces emissions by 21.4%')
    print('   - 2031-2050: No additional policies → emissions stabilize')
    print('   - This represents BAU (Business-as-Usual) after initial policies')
else:
    print('❌ UNEXPECTED: Emissions still changing after 2030')
    print(f'   2030→2040 change: {reduction_2040 - reduction_2030:.2f}%')
    print(f'   2030→2050 change: {reduction_2050 - reduction_2030:.2f}%')

print()
print('🔍 SECTOR-SPECIFIC FORECAST:')
print('-' * 70)
print(f"{'Sector':<15} {'2030 (Data)':<15} {'2040 (Forecast)':<15} {'2050 (Forecast)':<15}")
print('-' * 70)
for action in c.policy_actions:
    r_2030 = action.get_reduction_for_year(2030, c.target_year)
    r_2040 = action.get_reduction_for_year(2040, c.target_year)
    r_2050 = action.get_reduction_for_year(2050, c.target_year)

    print(f"{action.sector:<15} {r_2030:>6.1f}%{'':<8} {r_2040:>6.1f}%{'':<8} {r_2050:>6.1f}%")

print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print('✅ PURE FORECAST MODE enables:')
print('   1. Predict emissions based on 2026-2030 policy data only')
print('   2. No extrapolation beyond 2030 (stabilizes at last data point)')
print('   3. Useful for "what if we do nothing after 2030" scenario')
print('   4. BAU baseline for comparing against ambitious targets')
print()
print('💡 To use:')
print('   - Set all sector reduction_target_pct = 0')
print('   - Set country target_reduction_pct = 0')
print('   - System will maintain emission levels from last policy data year')
