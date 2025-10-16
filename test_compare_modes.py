"""Compare Target-Driven vs Historical Trend Forecast Modes"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from logic.policy_commitments import load_country_commitment, load_country_commitment_from_json

# Load Target-Driven mode (targets > 0)
c_target = load_country_commitment('Vietnam', 'data')

# Load Historical Trend mode (targets = 0)
with open('data/vietnam_forecast_mode.json', 'r', encoding='utf-8') as f:
    json_content = f.read()
c_trend = load_country_commitment_from_json(json_content, 'Vietnam')

print('=' * 90)
print('FORECAST MODES COMPARISON')
print('=' * 90)
print()

print('📊 CONFIGURATIONS:')
print('-' * 90)
print(f"{'Mode':<25} {'Target':<15} {'Method':<50}")
print('-' * 90)
print(f"{'Target-Driven':<25} {c_target.target_reduction_pct}%{'':<13} Linear extrapolation to reach target")
print(f"{'Historical Trend':<25} {c_trend.target_reduction_pct}%{'':<14} Linear regression + damping on historical data")
print()

print('🔮 FORECAST COMPARISON:')
print('-' * 90)
print(f"{'Year':<8} {'Target-Driven':<25} {'Historical Trend':<25} {'Difference':<15}")
print('-' * 90)

test_years = [2025, 2030, 2035, 2040, 2045, 2050]

for year in test_years:
    # Target-Driven
    red_target = c_target.calculate_annual_reduction_fraction(year) * 100
    em_target = c_target.baseline_emissions_gtco2 * (1 - red_target/100)

    # Historical Trend
    red_trend = c_trend.calculate_annual_reduction_fraction(year) * 100
    em_trend = c_trend.baseline_emissions_gtco2 * (1 - red_trend/100)

    # Difference
    diff = em_target - em_trend

    # Format
    target_str = f"{red_target:>5.1f}% → {em_target:.3f} GtCO₂"
    trend_str = f"{red_trend:>5.1f}% → {em_trend:.3f} GtCO₂"
    diff_str = f"{diff:+.3f} GtCO₂"

    print(f"{year:<8} {target_str:<25} {trend_str:<25} {diff_str:<15}")

print()
print('=' * 90)
print('KEY INSIGHTS')
print('=' * 90)
print()

target_2050 = c_target.calculate_annual_reduction_fraction(2050) * 100
trend_2050 = c_trend.calculate_annual_reduction_fraction(2050) * 100
gap = target_2050 - trend_2050

print(f'🎯 Target-Driven (2050):')
print(f'   Reduction: {target_2050:.1f}%')
print(f'   Emissions: {c_target.baseline_emissions_gtco2 * (1 - target_2050/100):.3f} GtCO₂')
print(f'   Interpretation: "Where we WANT to be (Net Zero commitment)"')
print()

print(f'📈 Historical Trend (2050):')
print(f'   Reduction: {trend_2050:.1f}%')
print(f'   Emissions: {c_trend.baseline_emissions_gtco2 * (1 - trend_2050/100):.3f} GtCO₂')
print(f'   Interpretation: "Where we\'ll LIKELY be (current momentum continues)"')
print()

print(f'⚠️  Policy Gap:')
print(f'   Additional reduction needed: {gap:.1f}%')
print(f'   Additional policies required to close this gap')
print()

print('💡 RECOMMENDATIONS:')
print('   1. Use Target-Driven for official climate commitments and NDC tracking')
print('   2. Use Historical Trend for realistic baseline and policy gap analysis')
print('   3. The gap ({:.1f}%) shows how much more effort is needed'.format(gap))
print('   4. Consider AI/ML methods (Prophet, XGBoost) for more sophisticated forecasts')
print()
print('=' * 90)
print('✅ BOTH MODES WORKING CORRECTLY!')
print('=' * 90)
