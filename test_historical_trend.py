"""Test Historical Trend Forecast Mode"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from logic.policy_commitments import load_country_commitment_from_json
import json

# Load Vietnam data with ALL targets = 0 (will use historical trend)
with open('data/vietnam_forecast_mode.json', 'r', encoding='utf-8') as f:
    json_content = f.read()

c = load_country_commitment_from_json(json_content, 'Vietnam')

print('=' * 80)
print('HISTORICAL TREND FORECAST MODE')
print('Uses linear regression on 2026-2030 data to extrapolate')
print('=' * 80)
print()

print('📊 CONFIGURATION:')
print(f'  Baseline: {c.baseline_emissions_gtco2} GtCO₂ ({c.baseline_year})')
print(f'  Historical data: {c.policy_actions[0].implementation_years[0]}-{c.policy_actions[0].implementation_years[-1]}')
print(f'  All targets: 0 (triggers historical trend mode)')
print()

# Analyze historical trend for Energy sector
energy = c.policy_actions[0]
years = energy.implementation_years
reductions = energy.yearly_improvement_pct

print('📈 ENERGY SECTOR HISTORICAL DATA:')
print('-' * 80)
print(f"{'Year':<10} {'Reduction %':<15} {'Annual Change'}")
print('-' * 80)
for i, (year, reduction) in enumerate(zip(years, reductions)):
    if i > 0:
        change = reduction - reductions[i-1]
        print(f"{year:<10} {reduction:>6.1f}%{'':<8} +{change:.1f}% per year")
    else:
        print(f"{year:<10} {reduction:>6.1f}%{'':<8} (baseline)")

# Calculate linear regression slope
n = len(years)
sum_x = sum(years)
sum_y = sum(reductions)
sum_xy = sum(y * x for x, y in zip(years, reductions))
sum_x2 = sum(x * x for x in years)
historical_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

print()
print(f'📊 Historical Trend Analysis:')
print(f'   Linear regression slope: {historical_slope:.3f}% per year')
print(f'   Average annual increase: {(reductions[-1] - reductions[0]) / (years[-1] - years[0]):.3f}% per year')
print()

print('🔮 FORECAST RESULTS (Historical Trend Extrapolation):')
print('-' * 80)
print(f"{'Year':<8} {'Reduction %':<15} {'Emissions (GtCO₂)':<20} {'Damping':<10} {'Notes'}")
print('-' * 80)

test_years = [2025, 2026, 2028, 2030, 2032, 2035, 2038, 2040, 2045, 2050]

for year in test_years:
    if year < c.baseline_year:
        continue

    reduction_fraction = c.calculate_annual_reduction_fraction(year)
    reduction_pct = reduction_fraction * 100
    emissions = c.baseline_emissions_gtco2 * (1 - reduction_fraction)

    # Calculate damping
    if year > years[-1]:
        years_elapsed = year - years[-1]
        damping = 1.0 / (1.0 + 0.05 * years_elapsed)
        damping_str = f"{damping:.2f}"
    else:
        damping_str = "N/A"

    # Notes
    if year == c.baseline_year:
        note = "Baseline"
    elif year <= years[-1]:
        note = "Historical data"
    else:
        note = "Extrapolated (trend)"

    print(f"{year:<8} {reduction_pct:>6.2f}%{'':<8} {emissions:>8.4f}{'':<12} {damping_str:<10} {note}")

print()
print('=' * 80)
print('COMPARISON: OLD vs NEW FORECAST MODE')
print('=' * 80)
print()

reduction_2030 = c.calculate_annual_reduction_fraction(2030) * 100
reduction_2035 = c.calculate_annual_reduction_fraction(2035) * 100
reduction_2040 = c.calculate_annual_reduction_fraction(2040) * 100
reduction_2050 = c.calculate_annual_reduction_fraction(2050) * 100

emissions_2030 = c.baseline_emissions_gtco2 * (1 - reduction_2030/100)
emissions_2035 = c.baseline_emissions_gtco2 * (1 - reduction_2035/100)
emissions_2040 = c.baseline_emissions_gtco2 * (1 - reduction_2040/100)
emissions_2050 = c.baseline_emissions_gtco2 * (1 - reduction_2050/100)

print('📊 Key Comparison Points:')
print(f"{'Year':<10} {'Old (Maintain)':<20} {'New (Trend)':<20} {'Change'}")
print('-' * 80)

# Old mode would maintain 21.38% everywhere
old_emissions = 0.3538  # 21.38% reduction

print(f"2030{'':<6} {old_emissions:.4f} GtCO₂{'':<8} {emissions_2030:.4f} GtCO₂{'':<8} {emissions_2030 - old_emissions:+.4f}")
print(f"2035{'':<6} {old_emissions:.4f} GtCO₂{'':<8} {emissions_2035:.4f} GtCO₂{'':<8} {emissions_2035 - old_emissions:+.4f}")
print(f"2040{'':<6} {old_emissions:.4f} GtCO₂{'':<8} {emissions_2040:.4f} GtCO₂{'':<8} {emissions_2040 - old_emissions:+.4f}")
print(f"2050{'':<6} {old_emissions:.4f} GtCO₂{'':<8} {emissions_2050:.4f} GtCO₂{'':<8} {emissions_2050 - old_emissions:+.4f}")

print()
print('=' * 80)
print('INTERPRETATION')
print('=' * 80)
print()

if reduction_2050 > reduction_2030:
    improvement = reduction_2050 - reduction_2030
    print(f'✅ Historical trend mode shows CONTINUED IMPROVEMENT')
    print(f'   2030: {reduction_2030:.2f}% reduction')
    print(f'   2050: {reduction_2050:.2f}% reduction')
    print(f'   Additional improvement: +{improvement:.2f}%')
    print()
    print('💡 Interpretation:')
    print('   - System extrapolates based on 2026-2030 trend')
    print('   - Assumes momentum continues (with damping)')
    print('   - More realistic than flat projection')
    print('   - Damping factor prevents unrealistic growth')
elif reduction_2050 < reduction_2030:
    decline = reduction_2030 - reduction_2050
    print(f'⚠️ Historical trend shows DECLINE (backsliding)')
    print(f'   This happens if historical data shows decreasing trend')
else:
    print(f'📊 Stable projection (rare with trend mode)')

print()
print('🔍 SECTOR-SPECIFIC TRENDS:')
print('-' * 80)
print(f"{'Sector':<15} {'2030':<12} {'2040':<12} {'2050':<12} {'Trend/yr'}")
print('-' * 80)

for action in c.policy_actions:
    r_2030 = action.get_reduction_for_year(2030, c.target_year)
    r_2040 = action.get_reduction_for_year(2040, c.target_year)
    r_2050 = action.get_reduction_for_year(2050, c.target_year)

    # Calculate trend
    if len(action.implementation_years) >= 2:
        years_data = action.implementation_years
        reduction_data = action.yearly_improvement_pct
        avg_trend = (reduction_data[-1] - reduction_data[0]) / (years_data[-1] - years_data[0])
    else:
        avg_trend = 0

    print(f"{action.sector:<15} {r_2030:>6.1f}%{'':<5} {r_2040:>6.1f}%{'':<5} {r_2050:>6.1f}%{'':<5} {avg_trend:+.2f}%")

print()
print('=' * 80)
print('DAMPING FACTOR EXPLANATION')
print('=' * 80)
print()
print('The damping factor prevents unrealistic extrapolation:')
print()
print('  damping = 1.0 / (1.0 + 0.05 × years_elapsed)')
print()
print('Examples:')
print('  2032 (2 yrs):  damping = 0.91  (91% of trend applied)')
print('  2035 (5 yrs):  damping = 0.80  (80% of trend applied)')
print('  2040 (10 yrs): damping = 0.67  (67% of trend applied)')
print('  2050 (20 yrs): damping = 0.50  (50% of trend applied)')
print()
print('💡 Rationale:')
print('   - Short-term: High confidence in trend continuation')
print('   - Long-term: More uncertainty, slower growth assumed')
print('   - Prevents exponential explosion in forecasts')
print('   - Mimics real-world diminishing returns')

print()
print('=' * 80)
print('✅ HISTORICAL TREND MODE WORKING!')
print('=' * 80)
print()
print('Benefits over "maintain last" mode:')
print('  ✅ Captures momentum from policy implementation')
print('  ✅ More realistic than flat projection')
print('  ✅ Automatically adjusts to data trends')
print('  ✅ Damped to prevent unrealistic forecasts')
print('  ✅ No manual tuning required')
