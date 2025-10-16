"""Test auto-calculation of share_pct"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from logic.policy_commitments import load_country_commitment

# Load Vietnam data (has manual share_pct in JSON)
commitment = load_country_commitment('Vietnam', 'data')

print('=' * 80)
print('AUTO-CALCULATION TEST')
print('(Current JSON has manual share_pct, testing if auto-calc matches)')
print('=' * 80)
print()

# Calculate expected gross emissions
gross = sum(
    action.baseline_emissions_mtco2
    for action in commitment.policy_actions
    if action.baseline_emissions_mtco2 > 0
)
print(f'Gross emissions: {gross} MtCO2')
print()

print(f"{'Sector':<15} {'Baseline':<12} {'Manual Share':<15} {'Auto Share':<15} {'Match?'}")
print('-' * 80)

all_match = True

for action in commitment.policy_actions:
    manual_share = action._manual_share_pct if action._manual_share_pct is not None else 0.0
    auto_share = action.share_pct

    if action.baseline_emissions_mtco2 > 0:
        expected = (action.baseline_emissions_mtco2 / gross) * 100
    else:
        expected = 0.0

    match_manual = abs(auto_share - manual_share) < 0.01
    match_expected = abs(auto_share - expected) < 0.01

    status = "MATCH" if match_manual else f"DIFF ({abs(auto_share - manual_share):.2f}%)"

    if not match_expected:
        all_match = False

    print(f'{action.sector:<15} {action.baseline_emissions_mtco2:>8.1f} MtCO2  '
          f'{manual_share:>6.2f}%        '
          f'{auto_share:>6.2f}%        '
          f'{status}')

print()
print('=' * 80)
print('SUMMARY')
print('=' * 80)

if all_match:
    print('PASS: All auto-calculations match expected values!')
    print('      Manual share_pct can be safely removed from JSON.')
else:
    print('FAIL: Some calculations differ from expected.')

print()
print('Testing calculation with 2030 data:')
reduction_2030 = commitment.calculate_annual_reduction_fraction(2030)
print(f'Total reduction fraction (2030): {reduction_2030:.4f} ({reduction_2030*100:.2f}%)')
print()

print('PASS: Auto-calculation implementation works correctly!' if all_match else 'FAIL: Issues detected')
