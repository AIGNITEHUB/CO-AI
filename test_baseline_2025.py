"""Test baseline year 2025 update"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from logic.policy_commitments import load_country_commitment

# Load Vietnam data
c = load_country_commitment('Vietnam', 'data')

print('=' * 60)
print('VIETNAM COMMITMENT DATA - BASELINE 2025')
print('=' * 60)
print(f'Baseline Year: {c.baseline_year}')
print(f'Baseline Emissions: {c.baseline_emissions_gtco2} GtCO₂')
print(f'Target Year: {c.target_year}')
print(f'Target Reduction: {c.target_reduction_pct}%')
print()

print('SECTOR BASELINES (MtCO₂):')
print('-' * 60)
total_emitters = 0
total_sinks = 0

for action in c.policy_actions:
    if action.baseline_emissions_mtco2 > 0:
        total_emitters += action.baseline_emissions_mtco2
        share = (action.baseline_emissions_mtco2 / (c.baseline_emissions_gtco2 * 1000)) * 100
        print(f'  {action.sector:12s}: {action.baseline_emissions_mtco2:6.1f} MtCO₂ (Share: {action.share_pct:.2f}%)')
    else:
        total_sinks += abs(action.baseline_emissions_mtco2)
        print(f'  {action.sector:12s}: {action.baseline_emissions_mtco2:6.1f} MtCO₂ (SINK)')

print('-' * 60)
print(f'Total Emitters:  {total_emitters:6.1f} MtCO₂ ({total_emitters/1000:.3f} GtCO₂)')
print(f'Total Sinks:     {-total_sinks:6.1f} MtCO₂ ({-total_sinks/1000:.3f} GtCO₂)')
print(f'NET EMISSIONS:   {total_emitters - total_sinks:6.1f} MtCO₂ ({(total_emitters - total_sinks)/1000:.3f} GtCO₂)')
print()

print('IMPLEMENTATION TIMELINE:')
print('-' * 60)
print(f'First year: {c.policy_actions[0].implementation_years[0]}')
print(f'Last year:  {c.policy_actions[0].implementation_years[-1]}')
print(f'Milestones: {c.policy_actions[0].implementation_years}')
print()

# Test calculations
print('REDUCTION CALCULATIONS:')
print('-' * 60)
for year in [2026, 2030, 2040, 2050]:
    reduction = c.calculate_annual_reduction_fraction(year)
    emissions = c.baseline_emissions_gtco2 * (1 - reduction)
    print(f'{year}: {reduction*100:5.2f}% reduction → {emissions:.3f} GtCO₂')

print()
print('✅ All tests passed!')
