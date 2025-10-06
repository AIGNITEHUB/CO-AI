#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test catalyst-pathway matrix and optimal conditions functions"""

import sys
from logic.conversion import (
    is_valid_combination,
    generate_catalyst_pathway_matrix,
    get_optimal_conditions_table
)

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("=" * 70)
print("Testing Catalyst-Pathway Matrix Functions")
print("=" * 70)

# Test 1: Validation function
print("\n" + "-" * 70)
print("Test 1: Pathway-Catalyst Validation")
print("-" * 70)

test_cases = [
    ("Sabatier (methanation)", "Ni-based", True),
    ("Sabatier (methanation)", "Cu (electro)", False),
    ("Electroreduction (formate route)", "Cu (electro)", True),
    ("Electroreduction (formate route)", "Ni-based", False),
    ("Methanol synthesis", "Other", True),
]

for pathway, catalyst, expected in test_cases:
    result = is_valid_combination(pathway, catalyst)
    status = "PASS" if result == expected else "FAIL"
    print(f"[{status}] {pathway} + {catalyst}: {result} (expected {expected})")

# Test 2: Generate matrix
print("\n" + "-" * 70)
print("Test 2: Catalyst-Pathway Performance Matrix")
print("-" * 70)

try:
    matrix_df = generate_catalyst_pathway_matrix()
    print(f"Matrix shape: {matrix_df.shape}")
    print(f"Pathways: {len(matrix_df)}")
    print(f"Catalysts: {len(matrix_df.columns) - 1}")
    print("\nMatrix preview:")
    print(matrix_df.to_string())
    print("\nTest 2 PASSED")
except Exception as e:
    print(f"Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Optimal conditions table
print("\n" + "-" * 70)
print("Test 3: Optimal Operating Conditions Table")
print("-" * 70)

try:
    conditions_df = get_optimal_conditions_table()
    print(f"Table shape: {conditions_df.shape}")
    print(f"Pathways: {len(conditions_df)}")
    print("\nConditions table preview:")
    print(conditions_df.to_string())

    # Show max and min yields
    max_idx = conditions_df['Max Yield (%)'].idxmax()
    min_idx = conditions_df['Max Yield (%)'].idxmin()

    print(f"\nHighest yield: {conditions_df.loc[max_idx, 'Pathway']} - {conditions_df.loc[max_idx, 'Max Yield (%)']}%")
    print(f"Lowest yield: {conditions_df.loc[min_idx, 'Pathway']} - {conditions_df.loc[min_idx, 'Max Yield (%)']}%")

    print("\nTest 3 PASSED")
except Exception as e:
    print(f"Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
