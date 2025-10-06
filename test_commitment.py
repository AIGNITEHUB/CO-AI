#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick test for commitment forecasting feature"""

import pandas as pd
import numpy as np
import sys
from logic.forecast import forecast_with_commitment

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Load sample data
df = pd.read_csv("data/sample_co2_global.csv")

print("=" * 60)
print("Testing Commitment Forecasting Feature")
print("=" * 60)

print(f"\nLoaded {len(df)} rows of historical data")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")
print(f"Current emissions: {df.iloc[-1]['emissions_gtco2']:.2f} GtCO2")

# Test 1: Net Zero by 2050 with exponential pathway
print("\n" + "-" * 60)
print("Test 1: Net Zero by 2050 (Exponential pathway)")
print("-" * 60)

try:
    result = forecast_with_commitment(
        df=df,
        target_year=2050,
        target_emissions=0.0,
        pathway_type='exponential',
        bau_degree=2
    )

    print(f"✓ BAU model: {result.bau_model_name}")
    print(f"✓ Pathway: {result.pathway_type}")
    print(f"✓ Forecast years: {len(result.future_years)}")
    print(f"✓ BAU 2050: {result.bau_forecast[-1]:.2f} GtCO2")
    print(f"✓ Commitment 2050: {result.commitment_forecast[-1]:.2f} GtCO2")
    print(f"✓ Total emissions gap: {np.sum(result.emissions_gap):.2f} GtCO2")
    print(f"✓ Avg annual reduction: {np.sum(result.emissions_gap)/len(result.future_years):.2f} GtCO2/yr")
    print("✓ Test 1 PASSED")
except Exception as e:
    print(f"✗ Test 1 FAILED: {e}")

# Test 2: 50% reduction by 2040 with linear pathway
print("\n" + "-" * 60)
print("Test 2: 50% reduction by 2040 (Linear pathway)")
print("-" * 60)

try:
    current = df.iloc[-1]['emissions_gtco2']
    target = current * 0.5

    result = forecast_with_commitment(
        df=df,
        target_year=2040,
        target_emissions=target,
        pathway_type='linear',
        bau_degree=2
    )

    print(f"✓ Target: {target:.2f} GtCO2 (50% of current)")
    print(f"✓ Commitment 2040: {result.commitment_forecast[-1]:.2f} GtCO2")
    print(f"✓ Difference from target: {abs(result.commitment_forecast[-1] - target):.4f} GtCO2")
    print("✓ Test 2 PASSED")
except Exception as e:
    print(f"✗ Test 2 FAILED: {e}")

# Test 3: S-curve pathway
print("\n" + "-" * 60)
print("Test 3: Net Zero by 2060 (S-curve pathway)")
print("-" * 60)

try:
    result = forecast_with_commitment(
        df=df,
        target_year=2060,
        target_emissions=0.0,
        pathway_type='scurve',
        bau_degree=2
    )

    print(f"✓ S-curve pathway generated")
    print(f"✓ Commitment 2060: {result.commitment_forecast[-1]:.2f} GtCO2")
    print("✓ Test 3 PASSED")
except Exception as e:
    print(f"✗ Test 3 FAILED: {e}")

# Test 4: Custom milestones
print("\n" + "-" * 60)
print("Test 4: Custom milestones pathway")
print("-" * 60)

try:
    milestones = {
        2030: 25.0,
        2040: 15.0,
    }

    result = forecast_with_commitment(
        df=df,
        target_year=2050,
        target_emissions=5.0,
        pathway_type='milestones',
        milestones=milestones,
        bau_degree=2
    )

    print(f"✓ Milestones: {milestones}")
    print(f"✓ Commitment 2050: {result.commitment_forecast[-1]:.2f} GtCO2")
    print("✓ Test 4 PASSED")
except Exception as e:
    print(f"✗ Test 4 FAILED: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
